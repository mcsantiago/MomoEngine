#include "Momo/Core/SceneFile.h"

#include <cmath>
#include <fstream>
#include <regex>
#include <sstream>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <nlohmann/json.hpp>

namespace Momo
{
    namespace
    {
        using nlohmann::json;

        // A malformed file, with the place in it that's wrong
        struct SceneFormatError
        {
            std::string message;
        };

        glm::vec3 ReadVec3(const json& value, const std::string& where)
        {
            if (!value.is_array() || value.size() != 3 ||
                !value[0].is_number() || !value[1].is_number() || !value[2].is_number())
                throw SceneFormatError{ where + " must be an array of 3 numbers" };
            return { value[0].get<float>(), value[1].get<float>(), value[2].get<float>() };
        }

        float ReadFloat(const json& value, const std::string& where)
        {
            if (!value.is_number())
                throw SceneFormatError{ where + " must be a number" };
            return value.get<float>();
        }

        template <typename T>
        std::optional<T> ReadOptional(const json& object, const char* key, const std::string& where, T (*read)(const json&, const std::string&))
        {
            auto it = object.find(key);
            if (it == object.end())
                return std::nullopt;
            return read(*it, where + "." + key);
        }

        const json& RequireObject(const json& value, const std::string& where)
        {
            if (!value.is_object())
                throw SceneFormatError{ where + " must be an object" };
            return value;
        }

        // Floats written with every digit read badly (34.99999237060547), so
        // values are rounded to a precision no one placing things by hand needs
        double Tidy(float value)
        {
            double rounded = std::round(static_cast<double>(value) * 1e5) / 1e5;
            return rounded == 0.0 ? 0.0 : rounded; // No "-0.0"
        }

        json WriteVec3(glm::vec3 value)
        {
            return json::array({ Tidy(value.x), Tidy(value.y), Tidy(value.z) });
        }

        SceneDescription ParseSceneJson(const json& root)
        {
            RequireObject(root, "The scene");
            if (auto version = root.find("version"); version != root.end())
            {
                if (!version->is_number_integer())
                    throw SceneFormatError{ "version must be an integer" };
                if (version->get<int>() > SceneFileVersion)
                    throw SceneFormatError{ "version " + std::to_string(version->get<int>()) + " is newer than this engine reads (" + std::to_string(SceneFileVersion) + ")" };
            }

            SceneDescription scene;
            if (auto camera = root.find("camera"); camera != root.end())
            {
                RequireObject(*camera, "camera");
                if (!camera->contains("position") || !camera->contains("target"))
                    throw SceneFormatError{ "camera needs a position and a target" };
                scene.camera = SceneCameraDescription{
                    .position = ReadVec3((*camera)["position"], "camera.position"),
                    .target = ReadVec3((*camera)["target"], "camera.target"),
                };
            }

            if (auto lighting = root.find("lighting"); lighting != root.end())
            {
                RequireObject(*lighting, "lighting");
                if (auto sun = lighting->find("sun"); sun != lighting->end())
                {
                    RequireObject(*sun, "lighting.sun");
                    if (!sun->contains("towardSun"))
                        throw SceneFormatError{ "lighting.sun needs towardSun" };
                    glm::vec3 toward = ReadVec3((*sun)["towardSun"], "lighting.sun.towardSun");
                    if (glm::length(toward) < 1e-6f)
                        throw SceneFormatError{ "lighting.sun.towardSun must not be zero" };
                    scene.sun = Assets::SunSource{
                        .towardSun = glm::normalize(toward),
                        .color = ReadOptional(*sun, "color", "lighting.sun", ReadVec3).value_or(glm::vec3(1.0f)),
                        .intensity = ReadOptional(*sun, "intensity", "lighting.sun", ReadFloat).value_or(1.0f),
                    };
                }
                if (auto sky = lighting->find("sky"); sky != lighting->end())
                {
                    RequireObject(*sky, "lighting.sky");
                    scene.sky.zenith = ReadOptional(*sky, "zenith", "lighting.sky", ReadVec3);
                    scene.sky.horizon = ReadOptional(*sky, "horizon", "lighting.sky", ReadVec3);
                    scene.sky.ground = ReadOptional(*sky, "ground", "lighting.sky", ReadVec3);
                    scene.sky.exposure = ReadOptional(*sky, "exposure", "lighting.sky", ReadFloat);
                }
            }

            auto entities = root.find("entities");
            if (entities == root.end() || !entities->is_array())
                throw SceneFormatError{ "The scene needs an \"entities\" array" };
            for (size_t i = 0; i < entities->size(); ++i)
            {
                std::string where = "entities[" + std::to_string(i) + "]";
                const json& entity = RequireObject((*entities)[i], where);
                auto asset = entity.find("asset");
                if (asset == entity.end() || !asset->is_string() || asset->get<std::string>().empty())
                    throw SceneFormatError{ where + " needs an \"asset\" path" };

                EntityDescription description;
                description.asset = std::filesystem::path(asset->get<std::string>());
                if (auto name = entity.find("name"); name != entity.end())
                {
                    if (!name->is_string())
                        throw SceneFormatError{ where + ".name must be a string" };
                    description.name = name->get<std::string>();
                }
                description.position = ReadOptional(entity, "position", where, ReadVec3).value_or(description.position);
                description.rotation = ReadOptional(entity, "rotation", where, ReadVec3).value_or(description.rotation);
                description.scale = ReadOptional(entity, "scale", where, ReadVec3).value_or(description.scale);
                if (auto isStatic = entity.find("static"); isStatic != entity.end())
                {
                    if (!isStatic->is_boolean())
                        throw SceneFormatError{ where + ".static must be true or false" };
                    description.isStatic = isStatic->get<bool>();
                }
                scene.entities.push_back(std::move(description));
            }
            return scene;
        }
    } // namespace

    glm::mat4 EntityDescription::Transform() const
    {
        return ComposeTransform(position, rotation, scale);
    }

    std::optional<SceneDescription> ParseScene(const std::string& text, std::string& error)
    {
        try
        {
            return ParseSceneJson(json::parse(text));
        }
        catch (const json::exception& e)
        {
            error = e.what();
        }
        catch (const SceneFormatError& e)
        {
            error = e.message;
        }
        return std::nullopt;
    }

    std::optional<SceneDescription> ReadSceneFile(const std::filesystem::path& path, std::string& error)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file)
        {
            error = "Can't open " + path.generic_string();
            return std::nullopt;
        }
        std::stringstream text;
        text << file.rdbuf();
        std::optional<SceneDescription> scene = ParseScene(text.str(), error);
        if (!scene)
            error = path.generic_string() + ": " + error;
        return scene;
    }

    std::string SerializeScene(const SceneDescription& scene)
    {
        // nlohmann's ordered_json keeps keys in the order written, so files
        // read top to bottom: version, camera, lighting, then the entities
        nlohmann::ordered_json root;
        root["version"] = SceneFileVersion;
        if (scene.camera)
            root["camera"] = { { "position", WriteVec3(scene.camera->position) }, { "target", WriteVec3(scene.camera->target) } };

        nlohmann::ordered_json lighting = nlohmann::ordered_json::object();
        if (scene.sun)
            lighting["sun"] = { { "towardSun", WriteVec3(scene.sun->towardSun) },
                                { "color", WriteVec3(scene.sun->color) },
                                { "intensity", Tidy(scene.sun->intensity) } };
        nlohmann::ordered_json sky = nlohmann::ordered_json::object();
        if (scene.sky.zenith) sky["zenith"] = WriteVec3(*scene.sky.zenith);
        if (scene.sky.horizon) sky["horizon"] = WriteVec3(*scene.sky.horizon);
        if (scene.sky.ground) sky["ground"] = WriteVec3(*scene.sky.ground);
        if (scene.sky.exposure) sky["exposure"] = Tidy(*scene.sky.exposure);
        if (!sky.empty())
            lighting["sky"] = sky;
        if (!lighting.empty())
            root["lighting"] = lighting;

        nlohmann::ordered_json entities = nlohmann::ordered_json::array();
        for (const EntityDescription& entity : scene.entities)
        {
            nlohmann::ordered_json written;
            if (!entity.name.empty())
                written["name"] = entity.name;
            written["asset"] = entity.asset.generic_string();
            written["position"] = WriteVec3(entity.position);
            written["rotation"] = WriteVec3(entity.rotation);
            written["scale"] = WriteVec3(entity.scale);
            written["static"] = entity.isStatic;
            entities.push_back(std::move(written));
        }
        root["entities"] = std::move(entities);
        // nlohmann puts every array element on a line of its own; vectors
        // read far better on one. (Strings can't match: their newlines are escaped.)
        static const std::regex vector3(R"(\[\s*(-?[0-9.eE+-]+),\s*(-?[0-9.eE+-]+),\s*(-?[0-9.eE+-]+)\s*\])");
        return std::regex_replace(root.dump(2), vector3, "[$1, $2, $3]") + "\n";
    }

    bool WriteSceneFile(const SceneDescription& scene, const std::filesystem::path& path, std::string& error)
    {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        if (!file)
        {
            error = "Can't write " + path.generic_string();
            return false;
        }
        file << SerializeScene(scene);
        if (!file)
        {
            error = "Failed writing " + path.generic_string();
            return false;
        }
        return true;
    }

    glm::mat4 ComposeTransform(glm::vec3 position, glm::vec3 rotationDegrees, glm::vec3 scale)
    {
        glm::vec3 r = glm::radians(rotationDegrees);
        // X first, then Y, then Z, about the world axes: Rz * Ry * Rx
        glm::mat4 rotation = glm::eulerAngleZYX(r.z, r.y, r.x);
        return glm::translate(glm::mat4(1.0f), position) * rotation * glm::scale(glm::mat4(1.0f), scale);
    }

    void DecomposeTransform(const glm::mat4& transform, glm::vec3& position, glm::vec3& rotationDegrees, glm::vec3& scale)
    {
        position = glm::vec3(transform[3]);
        scale = { glm::length(glm::vec3(transform[0])), glm::length(glm::vec3(transform[1])), glm::length(glm::vec3(transform[2])) };
        glm::mat4 rotation(1.0f);
        for (int axis = 0; axis < 3; ++axis)
            rotation[axis] = glm::vec4(scale[axis] > 0.0f ? glm::vec3(transform[axis]) / scale[axis] : glm::vec3(0.0f), 0.0f);
        float z, y, x;
        glm::extractEulerAngleZYX(rotation, z, y, x);
        glm::vec3 angles = glm::degrees(glm::vec3(x, y, z));
        // Every rotation has a second set of angles, (x + 180, 180 - y, z + 180):
        // a turn of 200 about Y comes out as (180, -20, 180). Keep whichever
        // set has the smaller angles, as a person would write it.
        auto wrap = [](float degrees) { return degrees - 360.0f * std::round(degrees / 360.0f); };
        glm::vec3 other(wrap(angles.x + 180.0f), wrap(180.0f - angles.y), wrap(angles.z + 180.0f));
        auto size = [](glm::vec3 a) { return std::abs(a.x) + std::abs(a.y) + std::abs(a.z); };
        rotationDegrees = size(other) + 1e-3f < size(angles) ? other : angles;
    }
}
