#include <doctest/doctest.h>

#include "Momo/Core/SceneFile.h"

using namespace Momo;

namespace
{
bool Near(glm::vec3 a, glm::vec3 b, float tolerance = 1e-4f)
{
    return glm::all(glm::lessThanEqual(glm::abs(a - b), glm::vec3(tolerance)));
}

bool Near(const glm::mat4& a, const glm::mat4& b, float tolerance = 1e-4f)
{
    for (int column = 0; column < 4; ++column)
        for (int row = 0; row < 4; ++row)
            if (std::abs(a[column][row] - b[column][row]) > tolerance)
                return false;
    return true;
}
} // namespace

TEST_CASE("A scene entity only needs an asset")
{
    std::string error;
    std::optional<SceneDescription> scene = ParseScene(R"({ "entities": [ { "asset": "Street.glb" } ] })", error);
    REQUIRE_MESSAGE(scene, error);
    CHECK_FALSE(scene->camera);
    CHECK_FALSE(scene->sun);
    CHECK_FALSE(scene->sky.exposure);
    REQUIRE(scene->entities.size() == 1);
    const EntityDescription& entity = scene->entities[0];
    CHECK(entity.asset == std::filesystem::path("Street.glb"));
    CHECK(entity.name.empty());
    CHECK(entity.isStatic);
    CHECK(Near(entity.Transform(), glm::mat4(1.0f)));
}

TEST_CASE("An entity without an asset is a marker")
{
    std::string error;
    std::optional<SceneDescription> scene = ParseScene(
        R"({ "entities": [ { "name": "PlayerStart", "position": [1, 0, 6], "rotation": [0, 90, 0] } ] })", error);
    REQUIRE_MESSAGE(scene, error);
    REQUIRE(scene->entities.size() == 1);
    CHECK_FALSE(scene->entities[0].asset);
    CHECK(scene->entities[0].name == "PlayerStart");
    CHECK_FALSE(scene->entities[0].isStatic);

    // Saved without an asset or a static flag, and read back the same
    std::string saved = SerializeScene(*scene);
    CHECK(saved.find("asset") == std::string::npos);
    CHECK(saved.find("static") == std::string::npos);
    std::optional<SceneDescription> again = ParseScene(saved, error);
    REQUIRE_MESSAGE(again, error);
    CHECK_FALSE(again->entities[0].asset);
    CHECK(Near(again->entities[0].Transform(), scene->entities[0].Transform()));
}

TEST_CASE("A scene survives saving and loading")
{
    SceneDescription scene;
    scene.camera = SceneCameraDescription{ .position = { 1.0f, 1.6f, 5.0f }, .target = { 0.0f, 1.0f, -3.0f } };
    scene.sun = Assets::SunSource{ .towardSun = glm::normalize(glm::vec3(0.5f, 0.8f, 0.3f)), .color = { 1.0f, 0.9f, 0.8f }, .intensity = 5.0f };
    scene.sky.zenith = glm::vec3(0.2f, 0.4f, 0.85f);
    scene.sky.exposure = 0.55f;
    scene.entities.push_back({ .name = "Street", .asset = "Market/Market.glb" });
    scene.entities.push_back({ .name = "Crate", .asset = "../Models/Crate.glb", .position = { 2.0f, 0.0f, 4.0f },
                               .rotation = { 10.0f, 35.0f, -20.0f }, .scale = { 1.0f, 2.0f, 1.0f }, .isStatic = false });

    std::string error;
    std::optional<SceneDescription> loaded = ParseScene(SerializeScene(scene), error);
    REQUIRE_MESSAGE(loaded, error);
    REQUIRE(loaded->camera);
    CHECK(Near(loaded->camera->position, scene.camera->position));
    CHECK(Near(loaded->camera->target, scene.camera->target));
    REQUIRE(loaded->sun);
    CHECK(Near(loaded->sun->towardSun, scene.sun->towardSun));
    CHECK(Near(loaded->sun->color, scene.sun->color));
    CHECK(loaded->sun->intensity == doctest::Approx(5.0f));
    REQUIRE(loaded->sky.zenith);
    CHECK(Near(*loaded->sky.zenith, *scene.sky.zenith));
    CHECK_FALSE(loaded->sky.horizon);
    CHECK(*loaded->sky.exposure == doctest::Approx(0.55f));

    REQUIRE(loaded->entities.size() == 2);
    for (size_t i = 0; i < 2; ++i)
    {
        const EntityDescription& original = scene.entities[i];
        const EntityDescription& entity = loaded->entities[i];
        CHECK(entity.name == original.name);
        CHECK(entity.asset == original.asset);
        CHECK(Near(entity.position, original.position));
        CHECK(Near(entity.rotation, original.rotation));
        CHECK(Near(entity.scale, original.scale));
        CHECK(entity.isStatic == original.isStatic);
    }
}

TEST_CASE("Euler rotations apply about X, then Y, then Z")
{
    // A quarter turn about X takes +Y to +Z; a quarter turn about Y then takes +Z to +X
    glm::mat4 transform = ComposeTransform(glm::vec3(0.0f), { 90.0f, 90.0f, 0.0f }, glm::vec3(1.0f));
    CHECK(Near(glm::vec3(transform * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f)), { 1.0f, 0.0f, 0.0f }));
}

TEST_CASE("Transforms decompose back into position, rotation and scale")
{
    glm::vec3 position{ 3.0f, -1.0f, 7.5f };
    glm::vec3 rotation{ 25.0f, -60.0f, 140.0f };
    glm::vec3 scale{ 0.5f, 2.0f, 1.25f };
    glm::mat4 transform = ComposeTransform(position, rotation, scale);

    glm::vec3 p, r, s;
    DecomposeTransform(transform, p, r, s);
    CHECK(Near(p, position));
    CHECK(Near(s, scale));
    // Euler angles aren't unique, so compare what they do rather than the angles
    CHECK(Near(ComposeTransform(p, r, s), transform));
}

TEST_CASE("Decomposed rotations use the smaller of the two equivalent angle sets")
{
    glm::vec3 p, r, s;
    DecomposeTransform(ComposeTransform(glm::vec3(0.0f), { 0.0f, 200.0f, 0.0f }, glm::vec3(1.0f)), p, r, s);
    CHECK(Near(r, { 0.0f, -160.0f, 0.0f }, 1e-3f));
}

TEST_CASE("Saved scenes write vectors on one line")
{
    SceneDescription scene;
    scene.entities.push_back({ .name = "Crate", .asset = "Crate.glb", .position = { 1.5f, -2.0f, 3.0f } });
    CHECK(SerializeScene(scene).find(R"("position": [1.5, -2.0, 3.0])") != std::string::npos);
}

TEST_CASE("Malformed scenes are rejected with the place that's wrong")
{
    std::string error;
    CHECK_FALSE(ParseScene("{ not json", error));
    CHECK_FALSE(error.empty());

    CHECK_FALSE(ParseScene(R"({ "camera": { "position": [0, 1, 0], "target": [0, 0, 0] } })", error));
    CHECK(error.find("entities") != std::string::npos);

    CHECK_FALSE(ParseScene(R"({ "entities": [ { "asset": "" } ] })", error));
    CHECK(error.find("entities[0]") != std::string::npos);

    CHECK_FALSE(ParseScene(R"({ "entities": [ { "asset": "a.glb", "position": [1, 2] } ] })", error));
    CHECK(error.find("entities[0].position") != std::string::npos);

    CHECK_FALSE(ParseScene(R"({ "version": 99, "entities": [] })", error));
    CHECK(error.find("newer") != std::string::npos);
}
