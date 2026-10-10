#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include "Momo/Renderer/VulkanRenderer.h"

namespace Momo {

struct BenchmarkSpecification
{
    std::filesystem::path OutputPath;
    // Per view. Covers the GPU timing readback lag (timings arrive
    // MAX_FRAMES_IN_FLIGHT frames late) and lets clocks settle after a view change.
    uint32_t WarmupFrames = 60;
    uint32_t MeasuredFrames = 240;
};

struct BenchmarkView
{
    std::string Name;
    glm::vec3 Position;
    float Yaw;   // Degrees
    float Pitch; // Degrees
};

// What the results were measured on, recorded alongside them
struct BenchmarkEnvironment
{
    std::string Gpu;
    uint32_t Width;
    uint32_t Height;
    uint32_t Bounces;
};

// Renders a fixed list of camera views in ray traced mode, records the GPU
// time of the ray tracing dispatch for each, and writes the results as JSON
// (the format tools/publish_benchmark.py adds to the website's data file).
class Benchmark
{
public:
    explicit Benchmark(BenchmarkSpecification spec);

    // The view to render this frame; only valid while !IsFinished()
    const BenchmarkView& CurrentView() const { return m_Views[m_ViewIndex]; }
    bool IsFinished() const { return m_ViewIndex >= m_Views.size(); }

    // Call once per ray traced frame, after rendering it. Returns the index of
    // the view this frame finished, if any; its image is converged by then.
    std::optional<size_t> Record(const Renderer::DrawFrameTime& frameTime);
    // Where to save the screenshot of a view: next to the results file
    std::filesystem::path ScreenshotPath(size_t viewIndex) const;
    void SetScreenshotSaved(size_t viewIndex) { m_Samples[viewIndex].screenshotSaved = true; }
    bool WriteResults(const BenchmarkEnvironment& environment) const;

private:
    struct ViewSamples
    {
        std::vector<float> dispatchMs;
        std::vector<float> frameMs;
        bool screenshotSaved = false;
    };

    BenchmarkSpecification m_Spec;
    std::vector<BenchmarkView> m_Views;
    std::vector<ViewSamples> m_Samples;
    size_t m_ViewIndex = 0;
    uint32_t m_FramesInView = 0;
};

} // namespace Momo
