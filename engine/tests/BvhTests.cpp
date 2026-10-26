#include <doctest/doctest.h>

#include <algorithm>
#include <limits>
#include <random>
#include <vector>
#include "Momo/RayTracing/Bvh.h"

using namespace Momo::RayTracing;

namespace {

constexpr float Infinity = std::numeric_limits<float>::infinity();

TrianglePositions MakeTriangle(glm::vec3 a, glm::vec3 b, glm::vec3 c)
{
    return { glm::vec4(a, 0.0f), glm::vec4(b, 0.0f), glm::vec4(c, 0.0f) };
}

// Triangles scattered through a cube, sized like scene geometry relative to the
// cube: small and overlapping, so rays regularly pass through several leaves
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

bool Contains(const BvhNode& outer, const BvhNode& inner)
{
    return glm::all(glm::lessThanEqual(outer.boundsMin, inner.boundsMin)) &&
           glm::all(glm::greaterThanEqual(outer.boundsMax, inner.boundsMax));
}

bool ContainsPoint(const BvhNode& node, const glm::vec4& point)
{
    return glm::all(glm::lessThanEqual(node.boundsMin, glm::vec3(point))) &&
           glm::all(glm::greaterThanEqual(node.boundsMax, glm::vec3(point)));
}

// Walks the tree checking the structural invariants the shader relies on and
// returns how many times each (reordered) triangle is referenced by a leaf
std::vector<uint32_t> CheckStructure(const Bvh& bvh, std::span<const TrianglePositions> ordered, uint32_t maxDepth)
{
    std::vector<uint32_t> references(ordered.size(), 0);
    struct Entry { uint32_t node, depth; };
    std::vector<Entry> stack{ { 0, 0 } };
    while (!stack.empty())
    {
        auto [index, depth] = stack.back();
        stack.pop_back();
        REQUIRE(index < bvh.nodes.size());
        REQUIRE(depth <= maxDepth);
        const BvhNode& node = bvh.nodes[index];
        if (node.IsLeaf())
        {
            REQUIRE(node.leftOrFirst + node.triangleCount <= ordered.size());
            for (uint32_t i = node.leftOrFirst; i < node.leftOrFirst + node.triangleCount; i++)
            {
                references[i]++;
                CHECK(ContainsPoint(node, ordered[i].v0));
                CHECK(ContainsPoint(node, ordered[i].v1));
                CHECK(ContainsPoint(node, ordered[i].v2));
            }
        }
        else
        {
            REQUIRE(node.leftOrFirst + 1 < bvh.nodes.size());
            CHECK(Contains(node, bvh.nodes[node.leftOrFirst]));
            CHECK(Contains(node, bvh.nodes[node.leftOrFirst + 1]));
            stack.push_back({ node.leftOrFirst, depth + 1 });
            stack.push_back({ node.leftOrFirst + 1, depth + 1 });
        }
    }
    return references;
}

} // namespace

TEST_CASE("Ray/triangle intersection")
{
    TrianglePositions triangle = MakeTriangle({ 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 });

    SUBCASE("hit reports distance and barycentrics")
    {
        auto hit = IntersectTriangle({ { 0.25f, 0.5f, 2.0f }, { 0, 0, -1 } }, triangle, 0.0f, Infinity);
        REQUIRE(hit);
        CHECK(hit->t == doctest::Approx(2.0f));
        CHECK(hit->u == doctest::Approx(0.25f));
        CHECK(hit->v == doctest::Approx(0.5f));
    }
    SUBCASE("back face is hit too")
    {
        CHECK(IntersectTriangle({ { 0.25f, 0.25f, -2.0f }, { 0, 0, 1 } }, triangle, 0.0f, Infinity));
    }
    SUBCASE("unnormalized direction scales t")
    {
        auto hit = IntersectTriangle({ { 0.25f, 0.25f, 2.0f }, { 0, 0, -4 } }, triangle, 0.0f, Infinity);
        REQUIRE(hit);
        CHECK(hit->t == doctest::Approx(0.5f));
    }
    SUBCASE("misses outside the edges")
    {
        CHECK_FALSE(IntersectTriangle({ { 0.75f, 0.75f, 2.0f }, { 0, 0, -1 } }, triangle, 0.0f, Infinity));
        CHECK_FALSE(IntersectTriangle({ { -0.1f, 0.5f, 2.0f }, { 0, 0, -1 } }, triangle, 0.0f, Infinity));
    }
    SUBCASE("parallel ray misses")
    {
        CHECK_FALSE(IntersectTriangle({ { -1.0f, 0.25f, 0.0f }, { 1, 0, 0 } }, triangle, 0.0f, Infinity));
    }
    SUBCASE("respects the [tMin, tMax] interval")
    {
        Ray ray{ { 0.25f, 0.25f, 2.0f }, { 0, 0, -1 } };
        CHECK_FALSE(IntersectTriangle(ray, triangle, 0.0f, 1.5f));
        CHECK_FALSE(IntersectTriangle(ray, triangle, 2.5f, Infinity));
        CHECK_FALSE(IntersectTriangle({ { 0.25f, 0.25f, -2.0f }, { 0, 0, -1 } }, triangle, 0.0f, Infinity)); // behind the origin
    }
    SUBCASE("degenerate triangle never hits")
    {
        TrianglePositions line = MakeTriangle({ 0, 0, 0 }, { 1, 0, 0 }, { 2, 0, 0 });
        CHECK_FALSE(IntersectTriangle({ { 0.5f, 0.0f, 2.0f }, { 0, 0, -1 } }, line, 0.0f, Infinity));
    }
}

TEST_CASE("BVH over no triangles is empty")
{
    Bvh bvh = BuildBvh({});
    CHECK(bvh.nodes.empty());
    CHECK(bvh.triangleOrder.empty());
    CHECK_FALSE(IntersectBvh({ { 0, 0, 0 }, { 0, 0, 1 } }, bvh, {}, 0.0f, Infinity));
}

TEST_CASE("BVH over one triangle is a single leaf")
{
    std::vector<TrianglePositions> triangles{ MakeTriangle({ 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }) };
    Bvh bvh = BuildBvh(triangles);
    REQUIRE(bvh.nodes.size() == 1);
    CHECK(bvh.nodes[0].IsLeaf());
    CHECK(bvh.nodes[0].triangleCount == 1);
    CHECK(IntersectBvh({ { 0.25f, 0.25f, 1.0f }, { 0, 0, -1 } }, bvh, triangles, 0.0f, Infinity));
}

TEST_CASE("BVH structure is valid and references every triangle once")
{
    std::mt19937 rng(1234);
    std::vector<TrianglePositions> triangles = RandomTriangles(rng, 5000);
    BvhBuildSettings settings;
    Bvh bvh = BuildBvh(triangles, settings);

    REQUIRE(bvh.triangleOrder.size() == triangles.size());
    std::vector<uint32_t> sortedOrder = bvh.triangleOrder;
    std::sort(sortedOrder.begin(), sortedOrder.end());
    for (uint32_t i = 0; i < sortedOrder.size(); i++)
        REQUIRE(sortedOrder[i] == i);

    auto ordered = Reorder<TrianglePositions>(triangles, bvh.triangleOrder);
    std::vector<uint32_t> references = CheckStructure(bvh, ordered, settings.maxDepth);
    CHECK(std::all_of(references.begin(), references.end(), [](uint32_t count) { return count == 1; }));

    CHECK(bvh.stats.nodeCount == bvh.nodes.size());
    CHECK(bvh.stats.nodeCount == 2 * bvh.stats.leafCount - 1);
    CHECK(bvh.stats.maxDepth <= settings.maxDepth);
    CHECK(bvh.stats.maxLeafTriangles <= settings.maxLeafTriangles);
    CHECK(bvh.stats.sahCost > 0.0f);
}

TEST_CASE("BVH closest hit matches brute force")
{
    std::mt19937 rng(42);
    std::vector<TrianglePositions> triangles = RandomTriangles(rng, 2000);
    Bvh bvh = BuildBvh(triangles);
    auto ordered = Reorder<TrianglePositions>(triangles, bvh.triangleOrder);

    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    int hits = 0;
    for (int i = 0; i < 2000; i++)
    {
        // Origins both inside and outside the triangle cloud
        Ray ray{ glm::vec3(unit(rng), unit(rng), unit(rng)) * 15.0f,
                 glm::vec3(unit(rng), unit(rng), unit(rng)) };
        if (glm::dot(ray.direction, ray.direction) == 0.0f)
            continue;

        auto expected = IntersectBruteForce(ray, ordered, 0.0f, Infinity);
        auto actual = IntersectBvh(ray, bvh, ordered, 0.0f, Infinity);
        REQUIRE(expected.has_value() == actual.has_value());
        if (expected)
        {
            hits++;
            CHECK(actual->t == expected->t);
            CHECK(actual->triangle == expected->triangle);
        }
    }
    // Guard against the test passing vacuously because every ray missed
    CHECK(hits > 200);
}

TEST_CASE("Axis-aligned rays traverse correctly")
{
    // A grid of floor quads; rays straight down have zero x and z direction
    // components, and some start exactly on node bounds planes
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
    Bvh bvh = BuildBvh(triangles);
    auto ordered = Reorder<TrianglePositions>(triangles, bvh.triangleOrder);

    for (float x : { 0.0f, 0.3f, 4.0f, 7.5f, 15.9f, 16.0f })
    {
        for (float z : { 0.0f, 0.2f, 8.0f, 15.0f })
        {
            Ray ray{ { x, 5.0f, z }, { 0, -1, 0 } };
            auto expected = IntersectBruteForce(ray, ordered, 0.0f, Infinity);
            auto actual = IntersectBvh(ray, bvh, ordered, 0.0f, Infinity);
            REQUIRE(expected.has_value() == actual.has_value());
            if (expected)
                CHECK(actual->t == doctest::Approx(5.0f));
        }
    }
    CHECK(IntersectBvh({ { 3.5f, 5.0f, 3.5f }, { 0, -1, 0 } }, bvh, ordered, 0.0f, Infinity));
    CHECK_FALSE(IntersectBvh({ { 3.5f, 5.0f, 3.5f }, { 0, 1, 0 } }, bvh, ordered, 0.0f, Infinity));
}

TEST_CASE("Coincident triangles terminate with bounded leaves")
{
    // Every centroid is identical, so no SAH split exists; the median fallback
    // must still make progress and keep leaves within maxLeafTriangles
    std::vector<TrianglePositions> triangles(100, MakeTriangle({ 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }));
    BvhBuildSettings settings;
    Bvh bvh = BuildBvh(triangles, settings);
    CHECK(bvh.stats.maxLeafTriangles <= settings.maxLeafTriangles);
    CHECK(bvh.stats.maxDepth <= settings.maxDepth);
    auto ordered = Reorder<TrianglePositions>(triangles, bvh.triangleOrder);
    CHECK(IntersectBvh({ { 0.25f, 0.25f, 1.0f }, { 0, 0, -1 } }, bvh, ordered, 0.0f, Infinity));
}

TEST_CASE("Depth limit forces leaves")
{
    std::mt19937 rng(7);
    std::vector<TrianglePositions> triangles = RandomTriangles(rng, 1000);
    BvhBuildSettings settings;
    settings.maxDepth = 3;
    Bvh bvh = BuildBvh(triangles, settings);
    CHECK(bvh.stats.maxDepth <= 3);
    CHECK(bvh.stats.leafCount <= 8);
    auto ordered = Reorder<TrianglePositions>(triangles, bvh.triangleOrder);
    auto references = CheckStructure(bvh, ordered, settings.maxDepth);
    CHECK(std::all_of(references.begin(), references.end(), [](uint32_t count) { return count == 1; }));
}
