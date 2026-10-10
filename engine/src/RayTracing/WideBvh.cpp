#include "Momo/RayTracing/WideBvh.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace Momo::RayTracing {
namespace {

constexpr float Infinity = std::numeric_limits<float>::infinity();
constexpr uint32_t WideTraversalStackSize = 32;

float SurfaceArea(const BvhNode& node)
{
    glm::vec3 extent = node.boundsMax - node.boundsMin;
    return 2.0f * (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x);
}

// The slab test of Bvh.cpp, on an explicit box: NaN (a zero direction
// component with the origin on a slab plane) counts as unbounded
float IntersectBox(const Ray& ray, const glm::vec3& inverseDirection, const glm::vec3& boundsMin, const glm::vec3& boundsMax, float tMin, float tMax)
{
    float tEnter = tMin;
    float tExit = tMax;
    for (int axis = 0; axis < 3; axis++)
    {
        float t0 = (boundsMin[axis] - ray.origin[axis]) * inverseDirection[axis];
        float t1 = (boundsMax[axis] - ray.origin[axis]) * inverseDirection[axis];
        float tNear = std::min(std::isnan(t0) ? -Infinity : t0, std::isnan(t1) ? -Infinity : t1);
        float tFar = std::max(std::isnan(t0) ? Infinity : t0, std::isnan(t1) ? Infinity : t1);
        tEnter = std::max(tEnter, tNear);
        tExit = std::min(tExit, tFar);
    }
    return tEnter <= tExit ? tEnter : Infinity;
}

// The quantization step for a biased exponent: a float with exactly these exponent bits
float StepFromExponent(uint8_t exponent)
{
    return std::bit_cast<float>(uint32_t(exponent) << 23);
}

// Dequantization, exactly as the shader does it. The product is exact (an
// 8-bit integer times a power of two), so this rounds once whether or not the
// compiler fuses it into an FMA.
float Dequantize(float origin, uint8_t q, float step)
{
    return origin + float(q) * step;
}

// Direction of rays in an octant (bit i set: component i negative)
glm::vec3 OctantDirection(uint32_t octant)
{
    return { (octant & 1) ? -1.0f : 1.0f, (octant & 2) ? -1.0f : 1.0f, (octant & 4) ? -1.0f : 1.0f };
}

class WideBuilder
{
public:
    explicit WideBuilder(const Bvh& binary) : m_Binary(binary) {}

    WideBvh Build()
    {
        WideBvh wide;
        if (m_Binary.nodes.empty())
            return wide;

        m_Nodes.push_back({});
        std::vector<Pending> pending{ { 0, 0, 1 } };
        while (!pending.empty())
        {
            Pending next = pending.back();
            pending.pop_back();
            BuildNode(next, pending);
        }

        if (m_MaxDepth >= WideTraversalStackSize)
            throw std::runtime_error("Wide BVH depth " + std::to_string(m_MaxDepth) + " exceeds the traversal stack");

        wide.nodes = std::move(m_Nodes);
        wide.triangleOrder = std::move(m_Order);
        wide.stats.binary = m_Binary.stats;
        wide.stats.nodeCount = static_cast<uint32_t>(wide.nodes.size());
        wide.stats.maxDepth = m_MaxDepth;
        return wide;
    }

private:
    struct Pending { uint32_t binaryIndex, wideIndex, depth; };

    // The binary nodes that become one wide node's children: start from the
    // node's two children and keep replacing the largest interior one with
    // its own two children until there are 8 (or only leaves are left)
    std::vector<uint32_t> GatherChildren(uint32_t binaryIndex) const
    {
        const BvhNode& node = m_Binary.nodes[binaryIndex];
        if (node.IsLeaf())
            return { binaryIndex }; // Only the root can be a leaf: a one-leaf tree
        std::vector<uint32_t> children{ node.leftOrFirst, node.leftOrFirst + 1 };
        while (children.size() < 8)
        {
            int largest = -1;
            float largestArea = -1.0f;
            for (int i = 0; i < int(children.size()); i++)
            {
                const BvhNode& child = m_Binary.nodes[children[i]];
                if (!child.IsLeaf() && SurfaceArea(child) > largestArea)
                {
                    largest = i;
                    largestArea = SurfaceArea(child);
                }
            }
            if (largest < 0)
                break;
            uint32_t opened = children[largest];
            children[largest] = m_Binary.nodes[opened].leftOrFirst;
            children.push_back(m_Binary.nodes[opened].leftOrFirst + 1);
        }
        return children;
    }

    // Slot s is where rays of octant s start, so put there the child those
    // rays reach first (the one furthest back along their direction). Greedy
    // over all (child, slot) pairs, cheapest first.
    std::array<int, 8> AssignSlots(const std::vector<uint32_t>& children) const
    {
        glm::vec3 parentCenter(0.0f);
        for (uint32_t child : children)
            parentCenter += 0.5f * (m_Binary.nodes[child].boundsMin + m_Binary.nodes[child].boundsMax);
        parentCenter /= float(children.size());

        struct Candidate { float cost; int child; int slot; };
        std::vector<Candidate> candidates;
        for (int c = 0; c < int(children.size()); c++)
        {
            glm::vec3 center = 0.5f * (m_Binary.nodes[children[c]].boundsMin + m_Binary.nodes[children[c]].boundsMax);
            for (int s = 0; s < 8; s++)
                candidates.push_back({ glm::dot(center - parentCenter, OctantDirection(s)), c, s });
        }
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) { return a.cost < b.cost; });

        std::array<int, 8> slotChild;
        slotChild.fill(-1);
        std::vector<bool> placed(children.size(), false);
        for (const Candidate& candidate : candidates)
        {
            if (!placed[candidate.child] && slotChild[candidate.slot] < 0)
            {
                slotChild[candidate.slot] = candidate.child;
                placed[candidate.child] = true;
            }
        }
        return slotChild;
    }

    // Picks the smallest power-of-two step that spans the node in 255 steps
    // and quantizes every child's bounds outwards on that axis. Retries with
    // a coarser step if float rounding would leave a box short.
    static void QuantizeAxis(CompressedWideNode& node, int axis, const std::array<int, 8>& slotChild,
                             const std::vector<glm::vec3>& mins, const std::vector<glm::vec3>& maxs, float origin, float extent)
    {
        int exponent = extent > 0.0f ? int(std::ceil(std::log2(extent / 255.0f))) : -126;
        for (;; exponent++)
        {
            exponent = std::clamp(exponent, -126, 127);
            uint8_t biased = uint8_t(exponent + 127);
            float step = StepFromExponent(biased);
            bool fits = true;
            for (int s = 0; s < 8 && fits; s++)
            {
                if (slotChild[s] < 0)
                    continue;
                float low = mins[slotChild[s]][axis], high = maxs[slotChild[s]][axis];
                float qLow = std::clamp(std::floor((low - origin) / step), 0.0f, 255.0f);
                float qHigh = std::clamp(std::ceil((high - origin) / step), 0.0f, 255.0f);
                while (qLow > 0.0f && Dequantize(origin, uint8_t(qLow), step) > low)
                    qLow -= 1.0f;
                while (qHigh < 255.0f && Dequantize(origin, uint8_t(qHigh), step) < high)
                    qHigh += 1.0f;
                if (Dequantize(origin, uint8_t(qLow), step) > low || Dequantize(origin, uint8_t(qHigh), step) < high)
                {
                    fits = false;
                    break;
                }
                uint8_t* lows[3] = { node.lowX, node.lowY, node.lowZ };
                uint8_t* highs[3] = { node.highX, node.highY, node.highZ };
                lows[axis][s] = uint8_t(qLow);
                highs[axis][s] = uint8_t(qHigh);
            }
            if (fits)
            {
                node.exponent[axis] = biased;
                return;
            }
            if (exponent == 127)
                throw std::runtime_error("Wide BVH node too large to quantize");
        }
    }

    void BuildNode(const Pending& pending, std::vector<Pending>& queue)
    {
        m_MaxDepth = std::max(m_MaxDepth, pending.depth);
        std::vector<uint32_t> children = GatherChildren(pending.binaryIndex);
        std::array<int, 8> slotChild = AssignSlots(children);

        std::vector<glm::vec3> mins, maxs;
        glm::vec3 origin(Infinity), top(-Infinity);
        for (uint32_t child : children)
        {
            mins.push_back(m_Binary.nodes[child].boundsMin);
            maxs.push_back(m_Binary.nodes[child].boundsMax);
            origin = glm::min(origin, mins.back());
            top = glm::max(top, maxs.back());
        }

        CompressedWideNode node{};
        node.origin = origin;
        for (int axis = 0; axis < 3; axis++)
            QuantizeAxis(node, axis, slotChild, mins, maxs, origin[axis], top[axis] - origin[axis]);

        // Interior children get consecutive nodes in slot order; leaf triangles
        // are appended in slot order from triangleBase
        node.childBase = static_cast<uint32_t>(m_Nodes.size());
        node.triangleBase = static_cast<uint32_t>(m_Order.size());
        for (int s = 0; s < 8; s++)
        {
            if (slotChild[s] < 0)
                continue;
            uint32_t binaryChild = children[slotChild[s]];
            const BvhNode& child = m_Binary.nodes[binaryChild];
            if (!child.IsLeaf())
            {
                node.internalMask |= uint8_t(1u << s);
                node.meta[s] = WideInteriorMeta;
                uint32_t wideIndex = static_cast<uint32_t>(m_Nodes.size());
                m_Nodes.push_back({});
                queue.push_back({ binaryChild, wideIndex, pending.depth + 1 });
                continue;
            }
            if (child.triangleCount > WideLeafMaxTriangles)
                throw std::runtime_error("Binary leaf of " + std::to_string(child.triangleCount) + " triangles is too large for a wide BVH leaf");
            uint32_t offset = static_cast<uint32_t>(m_Order.size()) - node.triangleBase;
            assert(offset < 32);
            uint32_t unaryCount = (1u << child.triangleCount) - 1;
            node.meta[s] = uint8_t((unaryCount << 5) | offset);
            for (uint32_t i = child.leftOrFirst; i < child.leftOrFirst + child.triangleCount; i++)
                m_Order.push_back(m_Binary.triangleOrder[i]);
        }
        m_Nodes[pending.wideIndex] = node;
    }

    const Bvh& m_Binary;
    std::vector<CompressedWideNode> m_Nodes;
    std::vector<uint32_t> m_Order;
    uint32_t m_MaxDepth = 0;
};

} // namespace

BvhBuildSettings WideBvhBinarySettings()
{
    BvhBuildSettings settings;
    settings.maxLeafTriangles = WideLeafMaxTriangles;
    // Leaves larger than a wide leaf can hold only happen at the depth limit,
    // so allow far more binary depth than the wide tree will end up with
    settings.maxDepth = 64;
    return settings;
}

void DecodeChildBounds(const CompressedWideNode& node, uint32_t slot, glm::vec3& boundsMin, glm::vec3& boundsMax)
{
    const uint8_t* lows[3] = { node.lowX, node.lowY, node.lowZ };
    const uint8_t* highs[3] = { node.highX, node.highY, node.highZ };
    for (int axis = 0; axis < 3; axis++)
    {
        float step = StepFromExponent(node.exponent[axis]);
        boundsMin[axis] = Dequantize(node.origin[axis], lows[axis][slot], step);
        boundsMax[axis] = Dequantize(node.origin[axis], highs[axis][slot], step);
    }
}

uint32_t RayOctant(const glm::vec3& direction)
{
    return (direction.x < 0.0f ? 1u : 0u) | (direction.y < 0.0f ? 2u : 0u) | (direction.z < 0.0f ? 4u : 0u);
}

WideBvh BuildWideBvh(std::span<const TrianglePositions> triangles)
{
    auto start = std::chrono::steady_clock::now();
    Bvh binary = BuildBvh(triangles, WideBvhBinarySettings());
    WideBvh wide = WideBuilder(binary).Build();
    wide.stats.buildMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return wide;
}

std::optional<Hit> IntersectWideBvh(const Ray& ray, const WideBvh& bvh, std::span<const TrianglePositions> triangles, float tMin, float tMax)
{
    if (bvh.nodes.empty())
        return std::nullopt;

    glm::vec3 inverseDirection = 1.0f / ray.direction;
    uint32_t octant = RayOctant(ray.direction);
    std::optional<Hit> closest;

    // A group is a node's children still to visit: one bit per visiting
    // position k (slot k ^ octant), so the lowest set bit is the next nearest.
    // Descending pushes the rest of the group as a single entry, so the stack
    // never holds more than one entry per level.
    struct Group { uint32_t node; uint32_t hits; };
    Group stack[WideTraversalStackSize];
    uint32_t stackSize = 0;
    uint32_t nodeIndex = 0;

    while (true)
    {
        // Visit a node: test all its children's boxes at once
        const CompressedWideNode& node = bvh.nodes[nodeIndex];
        Group group{ nodeIndex, 0 };
        for (uint32_t k = 0; k < 8; k++)
        {
            uint32_t slot = k ^ octant;
            if (node.meta[slot] == 0)
                continue;
            glm::vec3 boundsMin, boundsMax;
            DecodeChildBounds(node, slot, boundsMin, boundsMax);
            if (IntersectBox(ray, inverseDirection, boundsMin, boundsMax, tMin, tMax) != Infinity)
                group.hits |= 1u << k;
        }

        // Work through groups, testing leaves in place, until an interior child
        // to descend into turns up (or nothing is left)
        bool descend = false;
        while (!descend)
        {
            while (group.hits != 0)
            {
                uint32_t k = std::countr_zero(group.hits);
                group.hits &= group.hits - 1;
                const CompressedWideNode& parent = bvh.nodes[group.node];
                uint32_t slot = k ^ octant;
                if ((parent.internalMask >> slot) & 1)
                {
                    if (group.hits != 0)
                    {
                        assert(stackSize < WideTraversalStackSize);
                        stack[stackSize++] = group;
                    }
                    nodeIndex = parent.childBase + std::popcount(uint32_t(parent.internalMask) & ((1u << slot) - 1));
                    descend = true;
                    break;
                }

                uint32_t first = parent.triangleBase + (parent.meta[slot] & 31u);
                uint32_t count = std::popcount(uint32_t(parent.meta[slot]) >> 5);
                for (uint32_t i = first; i < first + count; i++)
                {
                    if (auto hit = IntersectTriangle(ray, triangles[i], tMin, tMax))
                    {
                        closest = hit;
                        closest->triangle = i;
                        tMax = hit->t;
                    }
                }
            }
            if (descend)
                break;
            if (stackSize == 0)
                return closest;
            group = stack[--stackSize];
        }
    }
}

} // namespace Momo::RayTracing
