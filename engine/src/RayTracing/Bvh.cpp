#include "Momo/RayTracing/Bvh.h"

#include <algorithm>
#include <chrono>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>

namespace Momo::RayTracing {
namespace {

constexpr float Infinity = std::numeric_limits<float>::infinity();

struct Aabb {
    glm::vec3 min = glm::vec3(Infinity);
    glm::vec3 max = glm::vec3(-Infinity);

    void Grow(const glm::vec3& point)
    {
        min = glm::min(min, point);
        max = glm::max(max, point);
    }

    void Grow(const Aabb& other)
    {
        min = glm::min(min, other.min);
        max = glm::max(max, other.max);
    }

    bool IsEmpty() const { return min.x > max.x; }

    float SurfaceArea() const
    {
        if (IsEmpty())
            return 0.0f;
        glm::vec3 extent = max - min;
        return 2.0f * (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x);
    }
};

class BvhBuilder
{
public:
    BvhBuilder(std::span<const TrianglePositions> triangles, const BvhBuildSettings& settings)
        : m_Settings(settings)
    {
        m_TriangleBounds.reserve(triangles.size());
        m_Centroids.reserve(triangles.size());
        for (const TrianglePositions& triangle : triangles)
        {
            Aabb bounds;
            bounds.Grow(glm::vec3(triangle.v0));
            bounds.Grow(glm::vec3(triangle.v1));
            bounds.Grow(glm::vec3(triangle.v2));
            m_TriangleBounds.push_back(bounds);
            m_Centroids.push_back((glm::vec3(triangle.v0) + glm::vec3(triangle.v1) + glm::vec3(triangle.v2)) / 3.0f);
        }
    }

    Bvh Build()
    {
        Bvh bvh;
        const uint32_t triangleCount = static_cast<uint32_t>(m_Centroids.size());
        if (triangleCount == 0)
            return bvh;

        m_Order.resize(triangleCount);
        std::iota(m_Order.begin(), m_Order.end(), 0u);

        // A binary tree over N triangles has at most 2N - 1 nodes
        m_Nodes.reserve(2 * static_cast<size_t>(triangleCount) - 1);
        m_Nodes.push_back({});
        Subdivide(0, 0, triangleCount, 0);

        bvh.nodes = std::move(m_Nodes);
        bvh.triangleOrder = std::move(m_Order);
        bvh.stats = m_Stats;
        bvh.stats.nodeCount = static_cast<uint32_t>(bvh.nodes.size());
        bvh.stats.sahCost = ComputeSahCost(bvh.nodes);
        return bvh;
    }

private:
    struct Split {
        int      axis = -1; // -1 = no valid split found
        uint32_t bin = 0;   // Triangles in bins [0, bin] go left
        float    cost = Infinity;
    };

    uint32_t BinOf(const glm::vec3& centroid, int axis, const Aabb& centroidBounds) const
    {
        float extent = centroidBounds.max[axis] - centroidBounds.min[axis];
        float scale = static_cast<float>(m_Settings.binCount) / extent;
        auto bin = static_cast<uint32_t>((centroid[axis] - centroidBounds.min[axis]) * scale);
        return std::min(bin, m_Settings.binCount - 1);
    }

    // Costs are left unnormalized (not divided by the node's area) so that
    // zero-area nodes, e.g. a stack of coincident triangles, don't divide by zero.
    Split FindBestSplit(uint32_t first, uint32_t count, const Aabb& centroidBounds) const
    {
        struct Bin {
            Aabb     bounds;
            uint32_t count = 0;
        };

        Split best;
        const uint32_t binCount = m_Settings.binCount;
        std::vector<Bin> bins(binCount);
        std::vector<float> leftCosts(binCount);

        for (int axis = 0; axis < 3; axis++)
        {
            if (centroidBounds.max[axis] - centroidBounds.min[axis] <= 0.0f)
                continue;

            std::fill(bins.begin(), bins.end(), Bin{});
            for (uint32_t i = first; i < first + count; i++)
            {
                uint32_t triangle = m_Order[i];
                Bin& bin = bins[BinOf(m_Centroids[triangle], axis, centroidBounds)];
                bin.bounds.Grow(m_TriangleBounds[triangle]);
                bin.count++;
            }

            // Sweep from the left, then from the right, so each candidate split
            // between bin i and i + 1 is costed in O(binCount) total
            Aabb leftBounds;
            uint32_t leftCount = 0;
            for (uint32_t i = 0; i + 1 < binCount; i++)
            {
                leftBounds.Grow(bins[i].bounds);
                leftCount += bins[i].count;
                leftCosts[i] = leftCount == 0 ? Infinity : leftBounds.SurfaceArea() * leftCount;
            }

            Aabb rightBounds;
            uint32_t rightCount = 0;
            for (uint32_t i = binCount - 1; i > 0; i--)
            {
                rightBounds.Grow(bins[i].bounds);
                rightCount += bins[i].count;
                if (rightCount == 0 || leftCosts[i - 1] == Infinity)
                    continue;

                float cost = m_Settings.intersectionCost * (leftCosts[i - 1] + rightBounds.SurfaceArea() * rightCount);
                if (cost < best.cost)
                    best = Split{ axis, i - 1, cost };
            }
        }
        return best;
    }

    void MakeLeaf(uint32_t nodeIndex, uint32_t first, uint32_t count)
    {
        m_Nodes[nodeIndex].leftOrFirst = first;
        m_Nodes[nodeIndex].triangleCount = count;
        m_Stats.leafCount++;
        m_Stats.maxLeafTriangles = std::max(m_Stats.maxLeafTriangles, count);
    }

    void Subdivide(uint32_t nodeIndex, uint32_t first, uint32_t count, uint32_t depth)
    {
        Aabb bounds, centroidBounds;
        for (uint32_t i = first; i < first + count; i++)
        {
            bounds.Grow(m_TriangleBounds[m_Order[i]]);
            centroidBounds.Grow(m_Centroids[m_Order[i]]);
        }
        m_Nodes[nodeIndex].boundsMin = bounds.min;
        m_Nodes[nodeIndex].boundsMax = bounds.max;
        m_Stats.maxDepth = std::max(m_Stats.maxDepth, depth);

        if (count <= 1 || depth >= m_Settings.maxDepth)
        {
            MakeLeaf(nodeIndex, first, count);
            return;
        }

        Split split = FindBestSplit(first, count, centroidBounds);
        float area = bounds.SurfaceArea();
        float leafCost = m_Settings.intersectionCost * count * area;
        float splitCost = m_Settings.traversalCost * area + split.cost;

        uint32_t middle;
        if (split.axis >= 0 && (splitCost < leafCost || count > m_Settings.maxLeafTriangles))
        {
            auto middleIt = std::partition(m_Order.begin() + first, m_Order.begin() + first + count,
                [&](uint32_t triangle) { return BinOf(m_Centroids[triangle], split.axis, centroidBounds) <= split.bin; });
            middle = static_cast<uint32_t>(middleIt - m_Order.begin());
        }
        else if (count > m_Settings.maxLeafTriangles)
        {
            // No usable SAH split: every centroid landed in one bin, e.g. many
            // triangles sharing a centroid. Split at the median along the widest
            // centroid axis, which still guarantees progress.
            glm::vec3 extent = centroidBounds.max - centroidBounds.min;
            int axis = extent.x >= extent.y && extent.x >= extent.z ? 0 : (extent.y >= extent.z ? 1 : 2);
            middle = first + count / 2;
            std::nth_element(m_Order.begin() + first, m_Order.begin() + middle, m_Order.begin() + first + count,
                [&](uint32_t a, uint32_t b) { return m_Centroids[a][axis] < m_Centroids[b][axis]; });
        }
        else
        {
            MakeLeaf(nodeIndex, first, count);
            return;
        }

        // Children are allocated as a pair so the right child is always left + 1
        auto leftIndex = static_cast<uint32_t>(m_Nodes.size());
        m_Nodes.push_back({});
        m_Nodes.push_back({});
        m_Nodes[nodeIndex].leftOrFirst = leftIndex;
        m_Nodes[nodeIndex].triangleCount = 0;

        Subdivide(leftIndex, first, middle - first, depth + 1);
        Subdivide(leftIndex + 1, middle, first + count - middle, depth + 1);
    }

    float ComputeSahCost(const std::vector<BvhNode>& nodes) const
    {
        Aabb root{ nodes[0].boundsMin, nodes[0].boundsMax };
        float rootArea = root.SurfaceArea();
        if (rootArea <= 0.0f)
            return 0.0f;

        float cost = 0.0f;
        for (const BvhNode& node : nodes)
        {
            float area = Aabb{ node.boundsMin, node.boundsMax }.SurfaceArea() / rootArea;
            cost += node.IsLeaf() ? m_Settings.intersectionCost * node.triangleCount * area
                                  : m_Settings.traversalCost * area;
        }
        return cost;
    }

    const BvhBuildSettings& m_Settings;
    std::vector<Aabb> m_TriangleBounds;
    std::vector<glm::vec3> m_Centroids;
    std::vector<uint32_t> m_Order;
    std::vector<BvhNode> m_Nodes;
    BvhStats m_Stats;
};

// Returns the entry distance into the box, or infinity on a miss.
// inverseDirection is a plain 1 / direction, so zero components become
// infinities. When the origin also lies on that axis' slab plane, 0 * inf is
// NaN; a ray parallel to an axis is inside that slab for every t (or for none),
// so NaN is treated as unbounded: -inf for entry, +inf for exit. This keeps
// boundary hits, such as a ray running along a box face, from being lost.
// The shader mirrors this.
float IntersectAabb(const Ray& ray, const glm::vec3& inverseDirection, const BvhNode& node, float tMin, float tMax)
{
    float tEnter = tMin;
    float tExit = tMax;
    for (int axis = 0; axis < 3; axis++)
    {
        float t0 = (node.boundsMin[axis] - ray.origin[axis]) * inverseDirection[axis];
        float t1 = (node.boundsMax[axis] - ray.origin[axis]) * inverseDirection[axis];
        float tNear = std::min(std::isnan(t0) ? -Infinity : t0, std::isnan(t1) ? -Infinity : t1);
        float tFar = std::max(std::isnan(t0) ? Infinity : t0, std::isnan(t1) ? Infinity : t1);
        tEnter = std::max(tEnter, tNear);
        tExit = std::min(tExit, tFar);
    }
    return tEnter <= tExit ? tEnter : Infinity;
}

} // namespace

Bvh BuildBvh(std::span<const TrianglePositions> triangles, const BvhBuildSettings& settings)
{
    assert(settings.binCount >= 2);
    auto start = std::chrono::steady_clock::now();
    Bvh bvh = BvhBuilder(triangles, settings).Build();
    bvh.stats.buildMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return bvh;
}

// Möller-Trumbore. Conditions are written as !(in range) so NaNs from
// degenerate triangles or parallel rays are rejected rather than accepted.
std::optional<Hit> IntersectTriangle(const Ray& ray, const TrianglePositions& triangle, float tMin, float tMax)
{
    glm::vec3 v0(triangle.v0);
    glm::vec3 edge1 = glm::vec3(triangle.v1) - v0;
    glm::vec3 edge2 = glm::vec3(triangle.v2) - v0;

    glm::vec3 p = glm::cross(ray.direction, edge2);
    float determinant = glm::dot(edge1, p);
    if (determinant == 0.0f)
        return std::nullopt;
    float inverseDeterminant = 1.0f / determinant;

    glm::vec3 s = ray.origin - v0;
    float u = glm::dot(s, p) * inverseDeterminant;
    if (!(u >= 0.0f && u <= 1.0f))
        return std::nullopt;

    glm::vec3 q = glm::cross(s, edge1);
    float v = glm::dot(ray.direction, q) * inverseDeterminant;
    if (!(v >= 0.0f && u + v <= 1.0f))
        return std::nullopt;

    float t = glm::dot(edge2, q) * inverseDeterminant;
    if (!(t > tMin && t < tMax))
        return std::nullopt;

    return Hit{ t, 0, u, v };
}

std::optional<Hit> IntersectBruteForce(const Ray& ray, std::span<const TrianglePositions> triangles, float tMin, float tMax)
{
    std::optional<Hit> closest;
    for (uint32_t i = 0; i < triangles.size(); i++)
    {
        if (auto hit = IntersectTriangle(ray, triangles[i], tMin, tMax))
        {
            closest = hit;
            closest->triangle = i;
            tMax = hit->t;
        }
    }
    return closest;
}

std::optional<Hit> IntersectBvh(const Ray& ray, const Bvh& bvh, std::span<const TrianglePositions> triangles, float tMin, float tMax)
{
    if (bvh.nodes.empty())
        return std::nullopt;

    glm::vec3 inverseDirection = 1.0f / ray.direction;
    if (IntersectAabb(ray, inverseDirection, bvh.nodes[0], tMin, tMax) == Infinity)
        return std::nullopt;

    // Near-first traversal pushes at most one node per level, so the build's
    // maxDepth bounds the stack. 64 leaves headroom over the default of 32.
    constexpr uint32_t StackSize = 64;
    uint32_t stack[StackSize];
    uint32_t stackSize = 0;
    uint32_t nodeIndex = 0;
    std::optional<Hit> closest;

    while (true)
    {
        const BvhNode& node = bvh.nodes[nodeIndex];
        if (node.IsLeaf())
        {
            for (uint32_t i = node.leftOrFirst; i < node.leftOrFirst + node.triangleCount; i++)
            {
                if (auto hit = IntersectTriangle(ray, triangles[i], tMin, tMax))
                {
                    closest = hit;
                    closest->triangle = i;
                    tMax = hit->t;
                }
            }
        }
        else
        {
            uint32_t nearChild = node.leftOrFirst;
            uint32_t farChild = node.leftOrFirst + 1;
            float nearDistance = IntersectAabb(ray, inverseDirection, bvh.nodes[nearChild], tMin, tMax);
            float farDistance = IntersectAabb(ray, inverseDirection, bvh.nodes[farChild], tMin, tMax);
            if (farDistance < nearDistance)
            {
                std::swap(nearChild, farChild);
                std::swap(nearDistance, farDistance);
            }

            if (nearDistance != Infinity)
            {
                if (farDistance != Infinity)
                {
                    assert(stackSize < StackSize);
                    stack[stackSize++] = farChild;
                }
                nodeIndex = nearChild;
                continue;
            }
        }

        if (stackSize == 0)
            break;
        nodeIndex = stack[--stackSize];
    }
    return closest;
}

} // namespace Momo::RayTracing
