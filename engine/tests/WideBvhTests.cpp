#include <doctest/doctest.h>

#include <algorithm>
#include <bit>
#include <limits>
#include <random>
#include <vector>
#include "Momo/RayTracing/WideBvh.h"

using namespace Momo::RayTracing;

namespace {

constexpr float Infinity = std::numeric_limits<float>::infinity();

TrianglePositions MakeTriangle(glm::vec3 a, glm::vec3 b, glm::vec3 c)
{
    return { glm::vec4(a, 0.0f), glm::vec4(b, 0.0f), glm::vec4(c, 0.0f) };
}

std::vector<TrianglePositions> RandomTriangles(std::mt19937& rng, size_t count)
{
    std::uniform_real_distribution<float> position(-10.0f, 10.0f);
    std::uniform_real_distribution<float> offset(-1.0f, 1.0f);
    std::vector<TrianglePositions> triangles;
    for (size_t i = 0; i < count; i++)
    {
        glm::vec3 center(position(rng), position(rng), position(rng));
        triangles.push_back(MakeTriangle(
            center + glm::vec3(offset(rng), offset(rng), offset(rng)),
            center + glm::vec3(offset(rng), offset(rng), offset(rng)),
            center + glm::vec3(offset(rng), offset(rng), offset(rng))));
    }
    return triangles;
}

bool Inside(const glm::vec3& boundsMin, const glm::vec3& boundsMax, const glm::vec4& point)
{
    return glm::all(glm::lessThanEqual(boundsMin, glm::vec3(point))) && glm::all(glm::greaterThanEqual(boundsMax, glm::vec3(point)));
}

// Walks the tree checking what traversal relies on: every node reached once,
// child indices in range, leaves within their size limit, and every triangle
// inside every stored box on its path from the root (the intersection of the
// ancestors' boxes is carried down). Returns how often each stored triangle
// is referenced.
std::vector<uint32_t> CheckStructure(const WideBvh& bvh, std::span<const TrianglePositions> ordered)
{
    std::vector<uint32_t> references(ordered.size(), 0);
    std::vector<uint32_t> visits(bvh.nodes.size(), 0);
    struct Entry { uint32_t node; glm::vec3 clipMin, clipMax; };
    std::vector<Entry> stack{ { 0, glm::vec3(-Infinity), glm::vec3(Infinity) } };
    while (!stack.empty())
    {
        Entry entry = stack.back();
        stack.pop_back();
        REQUIRE(entry.node < bvh.nodes.size());
        visits[entry.node]++;
        const CompressedWideNode& node = bvh.nodes[entry.node];
        for (uint32_t slot = 0; slot < 8; slot++)
        {
            if (node.meta[slot] == 0)
            {
                CHECK(((node.internalMask >> slot) & 1) == 0);
                continue;
            }
            glm::vec3 boundsMin, boundsMax;
            DecodeChildBounds(node, slot, boundsMin, boundsMax);
            glm::vec3 clipMin = glm::max(entry.clipMin, boundsMin);
            glm::vec3 clipMax = glm::min(entry.clipMax, boundsMax);
            if ((node.internalMask >> slot) & 1)
            {
                uint32_t child = node.childBase + std::popcount(uint32_t(node.internalMask) & ((1u << slot) - 1));
                stack.push_back({ child, clipMin, clipMax });
                continue;
            }
            uint32_t first = node.triangleBase + (node.meta[slot] & 31u);
            uint32_t count = std::popcount(uint32_t(node.meta[slot]) >> 5);
            REQUIRE(count >= 1);
            REQUIRE(count <= WideLeafMaxTriangles);
            REQUIRE(first + count <= ordered.size());
            for (uint32_t i = first; i < first + count; i++)
            {
                references[i]++;
                CHECK(Inside(clipMin, clipMax, ordered[i].v0));
                CHECK(Inside(clipMin, clipMax, ordered[i].v1));
                CHECK(Inside(clipMin, clipMax, ordered[i].v2));
            }
        }
    }
    CHECK(std::all_of(visits.begin(), visits.end(), [](uint32_t v) { return v == 1; }));
    return references;
}

void CheckMatchesBruteForce(const std::vector<TrianglePositions>& triangles, std::mt19937& rng, int rayCount, float origins)
{
    WideBvh bvh = BuildWideBvh(triangles);
    auto ordered = Reorder<TrianglePositions>(triangles, bvh.triangleOrder);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    int hits = 0;
    for (int i = 0; i < rayCount; i++)
    {
        Ray ray{ glm::vec3(unit(rng), unit(rng), unit(rng)) * origins, glm::vec3(unit(rng), unit(rng), unit(rng)) };
        if (glm::dot(ray.direction, ray.direction) == 0.0f)
            continue;
        auto expected = IntersectBruteForce(ray, ordered, 0.0f, Infinity);
        auto actual = IntersectWideBvh(ray, bvh, ordered, 0.0f, Infinity);
        REQUIRE(expected.has_value() == actual.has_value());
        if (expected)
        {
            hits++;
            CHECK(actual->t == expected->t);
            CHECK(actual->triangle == expected->triangle);
        }
    }
    CHECK(hits > rayCount / 10);
}

} // namespace

TEST_CASE("Wide BVH over no triangles is empty")
{
    WideBvh bvh = BuildWideBvh({});
    CHECK(bvh.nodes.empty());
    CHECK_FALSE(IntersectWideBvh({ { 0, 0, 0 }, { 0, 0, 1 } }, bvh, {}, 0.0f, Infinity));
}

TEST_CASE("Wide BVH over one triangle is a root with one leaf")
{
    std::vector<TrianglePositions> triangles{ MakeTriangle({ 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }) };
    WideBvh bvh = BuildWideBvh(triangles);
    REQUIRE(bvh.nodes.size() == 1);
    CHECK(bvh.nodes[0].internalMask == 0);
    CHECK(std::count_if(std::begin(bvh.nodes[0].meta), std::end(bvh.nodes[0].meta), [](uint8_t m) { return m != 0; }) == 1);
    CHECK(IntersectWideBvh({ { 0.25f, 0.25f, 1.0f }, { 0, 0, -1 } }, bvh, triangles, 0.0f, Infinity));
}

TEST_CASE("Wide BVH structure is valid and references every triangle once")
{
    std::mt19937 rng(1234);
    std::vector<TrianglePositions> triangles = RandomTriangles(rng, 5000);
    WideBvh bvh = BuildWideBvh(triangles);

    REQUIRE(bvh.triangleOrder.size() == triangles.size());
    std::vector<uint32_t> sortedOrder = bvh.triangleOrder;
    std::sort(sortedOrder.begin(), sortedOrder.end());
    for (uint32_t i = 0; i < sortedOrder.size(); i++)
        REQUIRE(sortedOrder[i] == i);

    auto ordered = Reorder<TrianglePositions>(triangles, bvh.triangleOrder);
    auto references = CheckStructure(bvh, ordered);
    CHECK(std::all_of(references.begin(), references.end(), [](uint32_t count) { return count == 1; }));

    CHECK(bvh.stats.nodeCount == bvh.nodes.size());
    // Eight-wide: far fewer nodes and levels than the binary tree it came from
    CHECK(bvh.stats.nodeCount < bvh.stats.binary.nodeCount / 4);
    CHECK(bvh.stats.maxDepth < bvh.stats.binary.maxDepth);
}

TEST_CASE("Wide BVH closest hit matches brute force")
{
    std::mt19937 rng(42);
    CheckMatchesBruteForce(RandomTriangles(rng, 2000), rng, 2000, 15.0f);
}

TEST_CASE("Wide BVH handles axis-aligned rays")
{
    std::vector<TrianglePositions> triangles;
    for (int x = 0; x < 16; x++)
    {
        for (int z = 0; z < 16; z++)
        {
            glm::vec3 p(x, 0, z);
            triangles.push_back(MakeTriangle(p, p + glm::vec3(1, 0, 0), p + glm::vec3(0, 0, 1)));
            triangles.push_back(MakeTriangle(p + glm::vec3(1, 0, 0), p + glm::vec3(1, 0, 1), p + glm::vec3(0, 0, 1)));
        }
    }
    WideBvh bvh = BuildWideBvh(triangles);
    auto ordered = Reorder<TrianglePositions>(triangles, bvh.triangleOrder);
    for (float x : { 0.0f, 0.3f, 4.0f, 7.5f, 15.9f, 16.0f })
    {
        for (float z : { 0.0f, 0.2f, 8.0f, 15.0f })
        {
            Ray ray{ { x, 5.0f, z }, { 0, -1, 0 } };
            auto expected = IntersectBruteForce(ray, ordered, 0.0f, Infinity);
            auto actual = IntersectWideBvh(ray, bvh, ordered, 0.0f, Infinity);
            REQUIRE(expected.has_value() == actual.has_value());
            if (expected)
                CHECK(actual->t == doctest::Approx(5.0f));
        }
    }
}

TEST_CASE("Wide BVH over coincident triangles stays within leaf limits")
{
    std::vector<TrianglePositions> triangles(100, MakeTriangle({ 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }));
    WideBvh bvh = BuildWideBvh(triangles);
    auto ordered = Reorder<TrianglePositions>(triangles, bvh.triangleOrder);
    auto references = CheckStructure(bvh, ordered);
    CHECK(std::all_of(references.begin(), references.end(), [](uint32_t count) { return count == 1; }));
    CHECK(IntersectWideBvh({ { 0.25f, 0.25f, 1.0f }, { 0, 0, -1 } }, bvh, ordered, 0.0f, Infinity));
}

TEST_CASE("Wide BVH quantized boxes are tight to within a step")
{
    // A box far from the origin (where float precision is coarse) must still
    // decode to cover its triangles, with at most one step of slack per side
    std::vector<TrianglePositions> triangles;
    for (int i = 0; i < 64; i++)
    {
        glm::vec3 p(1000.0f + i * 0.37f, -500.0f + i * 0.11f, 250.0f - i * 0.23f);
        triangles.push_back(MakeTriangle(p, p + glm::vec3(0.1f, 0, 0), p + glm::vec3(0, 0.1f, 0.05f)));
    }
    WideBvh bvh = BuildWideBvh(triangles);
    auto ordered = Reorder<TrianglePositions>(triangles, bvh.triangleOrder);
    auto references = CheckStructure(bvh, ordered);
    CHECK(std::all_of(references.begin(), references.end(), [](uint32_t count) { return count == 1; }));
}
