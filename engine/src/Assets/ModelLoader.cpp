#include "Momo/Assets/ModelLoader.h"
#include "Momo/Logging/Logger.h"
#include <fastgltf/core.hpp>
#include <stb_image.h>
#include <fastgltf/tools.hpp>
#include <string>
#include <variant>

namespace Momo::Assets {
class FastGltfModelLoader : public IModelLoader {
private:
    fastgltf::math::fmat4x4 mapGLMToFastGltf(const glm::mat4 &mat) {
        fastgltf::math::fmat4x4 result;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                result[i][j] = mat[i][j];
            }
        }
        return result;
    }

    glm::mat4 mapFastGltfToGLM(const fastgltf::math::fmat4x4 &mat) {
        glm::mat4 result;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                result[i][j] = mat[i][j];
            }
        }
        return result;
    }

    // Every decode path funnels through here so the pixel layout is decided in
    // one place. Forcing 4 channels keeps uploads simple: RGB8 is not reliably
    // supported as a sampled format on Vulkan implementations, RGBA8 always is.
    static constexpr int RequiredChannels = 4;

    std::optional<TextureSource> MakeTextureSource(stbi_uc *pixels, int width, int height) {
        if (!pixels) {
            return std::nullopt;
        }

        TextureSource textureSource;
        textureSource.width = static_cast<uint32_t>(width);
        textureSource.height = static_cast<uint32_t>(height);
        textureSource.channels = RequiredChannels;
        textureSource.pixels.assign(pixels,
                                    pixels + static_cast<size_t>(width) * height * RequiredChannels);
        stbi_image_free(pixels);
        return textureSource;
    }

    std::optional<TextureSource> LoadTextureSourceFromFile(const std::filesystem::path &path) {
        int width = 0, height = 0, sourceChannels = 0;
        stbi_uc *pixels = stbi_load(path.string().c_str(), &width, &height, &sourceChannels, RequiredChannels);
        if (!pixels) {
            LOG_ERROR(TAG, "stb_image failed to decode '{}': {}", path.string(), stbi_failure_reason());
            return std::nullopt;
        }
        return MakeTextureSource(pixels, width, height);
    }

    std::optional<TextureSource> LoadTextureSourceFromMemory(const std::byte *bytes, size_t byteLength) {
        int width = 0, height = 0, sourceChannels = 0;
        stbi_uc *pixels = stbi_load_from_memory(reinterpret_cast<const stbi_uc *>(bytes),
                                                static_cast<int>(byteLength),
                                                &width, &height, &sourceChannels, RequiredChannels);
        if (!pixels) {
            LOG_ERROR(TAG, "stb_image failed to decode embedded image: {}", stbi_failure_reason());
            return std::nullopt;
        }
        return MakeTextureSource(pixels, width, height);
    }

    std::optional<TextureSource> LoadTextureSourceFromArray(const fastgltf::sources::Array &arraySource) {
        return LoadTextureSourceFromMemory(arraySource.bytes.data(), arraySource.bytes.size());
    }

    // A buffer view points into a buffer, whose own data is another variant —
    // hence the second visit.
    std::optional<TextureSource> LoadTextureSourceFromBufferView(const fastgltf::Asset &gltf,
                                                                 const fastgltf::sources::BufferView &bufferViewSource) {
        const auto &bufferView = gltf.bufferViews[bufferViewSource.bufferViewIndex];
        const auto &buffer = gltf.buffers[bufferView.bufferIndex];

        return std::visit<std::optional<TextureSource>>(fastgltf::visitor{
            [&](const fastgltf::sources::Array &arraySource) {
                return LoadTextureSourceFromMemory(arraySource.bytes.data() + bufferView.byteOffset,
                                                   bufferView.byteLength);
            },
            [&](const fastgltf::sources::ByteView &byteView) {
                return LoadTextureSourceFromMemory(byteView.bytes.data() + bufferView.byteOffset,
                                                   bufferView.byteLength);
            },
            [&](auto&) -> std::optional<TextureSource> {
                LOG_WARN(TAG, "Unsupported buffer source backing an image buffer view");
                return std::nullopt;
            }
        }, buffer.data);
    }

    // Decodes one glTF image. Images are file-level, so this runs once per
    // image no matter how many primitives end up referencing it.
    std::optional<TextureSource> LoadTextureSource(const fastgltf::Asset &gltf,
                                                   const fastgltf::Image &image,
                                                   const std::filesystem::path &baseDir) {
        return std::visit<std::optional<TextureSource>>(fastgltf::visitor{
            [&](const fastgltf::sources::URI &uriSource) {
                if (uriSource.fileByteOffset != 0) {
                    LOG_WARN(TAG, "Image URI with a byte offset is not supported");
                    return std::optional<TextureSource>{};
                }
                return LoadTextureSourceFromFile(baseDir / std::filesystem::path(uriSource.uri.fspath()));
            },
            [&](const fastgltf::sources::BufferView &bufferView) {
                return LoadTextureSourceFromBufferView(gltf, bufferView);
            },
            [&](const fastgltf::sources::Array &arraySource) {
                return LoadTextureSourceFromArray(arraySource);
            },
            [&](auto&) -> std::optional<TextureSource> {
                LOG_WARN(TAG, "Unsupported image source");
                return std::nullopt;
            }
        }, image.data);
    }

    // Walks material -> texture -> image to find the image backing the base
    // colour, so materials can reference ModelSource::textures by index.
    std::optional<uint32_t> FindBaseColorImageIndex(const fastgltf::Asset &gltf,
                                                    const fastgltf::Material &material) {
        const auto &baseColorTexture = material.pbrData.baseColorTexture;
        if (!baseColorTexture.has_value()) {
            return std::nullopt;
        }

        if (baseColorTexture->texCoordIndex != 0) {
            LOG_WARN(TAG, "Base color texture uses TEXCOORD_{}, only TEXCOORD_0 is supported",
                     baseColorTexture->texCoordIndex);
        }

        const auto &texture = gltf.textures[baseColorTexture->textureIndex];
        if (!texture.imageIndex.has_value()) {
            LOG_WARN(TAG, "Texture has no associated image");
            return std::nullopt;
        }
        // TODO: texture.samplerIndex is dropped here; revisit when TextureData
        // grows sampling parameters.
        return static_cast<uint32_t>(*texture.imageIndex);
    }

    Assets::AlphaMode MapAlphaMode(const fastgltf::AlphaMode &alphaMode) {
        switch (alphaMode) {
            case fastgltf::AlphaMode::Opaque:
                return Assets::AlphaMode::Opaque;
            case fastgltf::AlphaMode::Mask:
                return Assets::AlphaMode::Mask;
            case fastgltf::AlphaMode::Blend:
                return Assets::AlphaMode::Blend;
            default:
                LOG_WARN(TAG, "Unknown alpha mode: {}", static_cast<int>(alphaMode));
                return Assets::AlphaMode::Opaque;
        }
    }

    fastgltf::Parser parser;
    const std::string TAG = "FastGltfModelLoader";

public:
    FastGltfModelLoader()
        : parser(fastgltf::Extensions::KHR_lights_punctual | fastgltf::Extensions::KHR_materials_emissive_strength) {}
        
    std::optional<Assets::ModelSource> LoadModel(const std::filesystem::path &path) override {
        // Implementation for loading a model using FastGltf goes here
        if (path.empty()) {
            LOG_ERROR(TAG, "Path is empty: {}", path.string());
            return std::nullopt;
        }

        auto model = fastgltf::GltfDataBuffer::FromPath(path);
        if (!model) {
            LOG_ERROR(TAG, "Failed to create GLTF data buffer from path: {}", path.string());
            return std::nullopt;
        }

        auto options = fastgltf::Options::LoadExternalBuffers | fastgltf::Options::LoadExternalImages;
        auto asset = parser.loadGltf(model.get(), path.parent_path(), options);
        if (!asset) {
            LOG_ERROR(TAG, "Failed to load GLTF asset from path: {}", path.string());
            return std::nullopt;
        }

        size_t sceneIndex = asset->defaultScene.value_or(0);
        // TODO: Identity for now, should be replaced with actual scene transformation from the entity
        fastgltf::math::fmat4x4 sceneTransform = mapGLMToFastGltf(glm::mat4(1.0f));

        std::vector<MeshSource> meshSources;
        std::vector<TextureSource> texSources;
        std::vector<MaterialSource> matSources;
        std::vector<PointLight> pointLights;
        std::optional<CameraSource> camera;

        // Pass 1: images. Indices line up 1:1 with the glTF image list, so a
        // failed decode leaves an empty slot instead of shifting every index
        // after it.
        texSources.resize(asset->images.size());
        for (size_t i = 0; i < asset->images.size(); ++i) {
            if (auto textureSource = LoadTextureSource(asset.get(), asset->images[i], path.parent_path())) {
                LOG_INFO(TAG, "Loaded image {}: {}x{} ({} channels, {} bytes)", i,
                         textureSource->width, textureSource->height, textureSource->channels,
                         textureSource->pixels.size());
                texSources[i] = std::move(*textureSource);
            } else {
                LOG_WARN(TAG, "Failed to load image at index: {}", i);
            }
        }

        // Pass 2: materials, also 1:1 with the glTF material list.
        matSources.resize(asset->materials.size());
        for (size_t i = 0; i < asset->materials.size(); ++i) {
            const auto &material = asset->materials[i];
            matSources[i].baseColorFactor = glm::vec4(
                material.pbrData.baseColorFactor[0],
                material.pbrData.baseColorFactor[1],
                material.pbrData.baseColorFactor[2],
                material.pbrData.baseColorFactor[3]
            );
            matSources[i].doubleSided = material.doubleSided;
            matSources[i].baseColorTexture = FindBaseColorImageIndex(asset.get(), material);
            matSources[i].alphaMode = MapAlphaMode(material.alphaMode);
            matSources[i].alphaCutoff = material.alphaCutoff;
            matSources[i].emissive = glm::vec3(
                material.emissiveFactor[0],
                material.emissiveFactor[1],
                material.emissiveFactor[2]
            ) * static_cast<float>(material.emissiveStrength);
        }

        // Pass 3: geometry per primitive. Materials and images are already
        // loaded, so this only records the index and never bails out early.
        fastgltf::iterateSceneNodes(asset.get(), sceneIndex, sceneTransform, [&](fastgltf::Node &node, fastgltf::math::fmat4x4 nodeTransform) {
            auto &gltf = asset.get();
            if (node.cameraIndex.has_value() && !camera.has_value()) {
                // glTF cameras look down their local -Z. Projection settings are
                // ignored: the engine's field of view is fixed
                glm::mat4 toModel = mapFastGltfToGLM(nodeTransform);
                camera = CameraSource{
                    .position = glm::vec3(toModel[3]),
                    .forward = glm::normalize(glm::mat3(toModel) * glm::vec3(0.0f, 0.0f, -1.0f)),
                };
            }
            if (node.lightIndex.has_value()) {
                const fastgltf::Light &light = gltf.lights[*node.lightIndex];
                if (light.type == fastgltf::LightType::Point) {
                    glm::mat4 toModel = mapFastGltfToGLM(nodeTransform);
                    glm::vec3 color(light.color[0], light.color[1], light.color[2]);
                    // range is ignored: the light reaches everywhere, falling off as 1 / d^2
                    pointLights.push_back(PointLight{
                        .position = glm::vec3(toModel[3]),
                        .intensity = color * static_cast<float>(light.intensity),
                    });
                } else {
                    LOG_WARN(TAG, "Skipping light '{}': only point lights are supported", light.name);
                }
            }
            if (node.meshIndex.has_value()) {
                LOG_INFO(TAG, "Node has a mesh index: {}", node.meshIndex.value());
                for (const auto &primitive : gltf.meshes[node.meshIndex.value()].primitives) {
                    if (primitive.type != fastgltf::PrimitiveType::Triangles) {
                        LOG_WARN(TAG, "Unsupported primitive type: Primitive type is not triangles, skipping");
                        continue;
                    }
                    auto positionAttribute = primitive.findAttribute("POSITION");
                    if (positionAttribute == primitive.attributes.end()) {
                        LOG_WARN(TAG, "Primitive does not have POSITION attribute, skipping");
                        continue;
                    }
                    auto normalAttribute = primitive.findAttribute("NORMAL");
                    if (normalAttribute == primitive.attributes.end()) {
                        LOG_WARN(TAG, "Primitive does not have NORMAL attribute, skipping");
                        continue;
                    }

                    Assets::MeshSource meshSource = Assets::MeshSource();
                    Geometry::MeshData meshData = Geometry::MeshData();
                    meshSource.localTransform = mapFastGltfToGLM(nodeTransform);
                    auto& posAccessor = gltf.accessors[positionAttribute->accessorIndex];
                    meshData.vertices.resize(posAccessor.count);

                    fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(gltf, posAccessor,
                    [&](fastgltf::math::fvec3 pos, std::size_t index) {
                        meshData.vertices[index].position = glm::vec3(pos.x(), pos.y(), pos.z());
                    });

                    auto& normAccessor = gltf.accessors[normalAttribute->accessorIndex];
                    fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(gltf, normAccessor,
                    [&](fastgltf::math::fvec3 norm, std::size_t index) {
                        meshData.vertices[index].normal = glm::vec3(norm.x(), norm.y(), norm.z());
                    });

                    auto& indexAccessorIdx = primitive.indicesAccessor;
                    if (indexAccessorIdx.has_value()) {
                        auto &indexAccessor = gltf.accessors[*indexAccessorIdx];
                        meshData.indices.resize(indexAccessor.count);
                        fastgltf::iterateAccessorWithIndex<std::uint32_t>(gltf, indexAccessor,
                        [&](std::uint32_t indexValue, std::size_t index) {
                            meshData.indices[index] = indexValue;
                        });
                    }
                    // TODO: Support other TEXCOORDs
                    auto texCoordAttribute = primitive.findAttribute("TEXCOORD_0");
                    if (texCoordAttribute != primitive.attributes.end()) {
                        auto &texCoordAccessor = gltf.accessors[texCoordAttribute->accessorIndex];
                        fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec2>(gltf, texCoordAccessor,
                        [&](fastgltf::math::fvec2 uv, std::size_t index) {
                            meshData.vertices[index].texCoord = glm::vec2(uv.x(), uv.y());
                        });
                    }

                    if (primitive.materialIndex.has_value()) {
                        meshSource.materialIndex = static_cast<uint32_t>(*primitive.materialIndex);
                    }

                    meshSource.geometry = std::move(meshData);
                    meshSources.push_back(std::move(meshSource));
                }
            }
        });

        Assets::ModelSource modelSource = Assets::ModelSource();
        modelSource.meshes = std::move(meshSources);
        modelSource.materials = std::move(matSources);
        modelSource.textures = std::move(texSources);
        modelSource.pointLights = std::move(pointLights);
        modelSource.camera = camera;

        // Load the model using FastGltf here
        return modelSource;
    }

}; 

// Factory
IModelLoader* IModelLoader::CreateGltfModelLoader()
{
    return new FastGltfModelLoader();
}
} // namespace Momo::Assets
