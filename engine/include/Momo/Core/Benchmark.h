#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include "Momo/Renderer/VulkanRenderer.h"
#include "Momo/Profiling/FrameStats.h"

namespace Momo {

struct BenchmarkSpecification
{
    std::filesystem::path OutputPath;
    // Per view. Covers the GPU timing readback lag (timings arrive
    // MAX_FRAMES_IN_FLIGHT frames late) and lets clocks settle after a view change.
    uint32_t WarmupFrames = 60;
    uint32_t MeasuredFrames = 240;
    // After the timed frames, frames rendered with the shader variant that
    // counts traversal work (slower, so never timed)
    uint32_t CountingFrames = 4;
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
    // The ray tracing scene: optimizations that reshape the BVH should show up here too
    RayTracing::BvhStats Bvh;
    size_t Triangles;
    size_t GpuBytes;
    size_t NodeBytes; // sizeof(RayTracing::BvhNode), to turn node reads into bytes
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
    // True while the current view renders its counting frames
    bool IsCounting() const { return !IsFinished() && m_CountingFramesLeft > 0; }
    // True for the frame that records rays for the trace-only experiment,
    // rendered after the counting frames
    bool IsCapturing() const { return !IsFinished() && m_Capturing; }

    // Call once per ray traced frame, after rendering it. Returns the index of
    // the view this frame finished, if any; its image is converged by then,
    // and the renderer's traversal counters hold its counting frames.
    std::optional<size_t> Record(const Renderer::DrawFrameTime& drawTime, const Profiling::FrameTime& frameTime);
    // Where to save the screenshot of a view: next to the results file
    std::filesystem::path ScreenshotPath(size_t viewIndex) const;
    void SetScreenshotSaved(size_t viewIndex) { m_Samples[viewIndex].screenshotSaved = true; }
    // pixels: pixels per frame (the counts cover CountingFrames frames)
    void SetTraversalCounts(size_t viewIndex, const Renderer::TraversalCounts& counts, uint64_t pixels);
    void SetTraceOnlyResults(size_t viewIndex, std::vector<Renderer::TraceOnlyQueueResult> results) { m_Samples[viewIndex].traceOnly = std::move(results); }
    bool WriteResults(const BenchmarkEnvironment& environment) const;

private:
    struct ViewSamples
    {
        std::vector<float> dispatchMs;
        std::vector<float> frameMs;
        // CPU side. While GPU-bound these mostly measure waiting on the GPU
        // (fence wait ~ GPU time), but they show if that ever stops being true.
        std::vector<float> cpuTotalMs;
        std::vector<float> recordMs;
        std::vector<float> submitMs; // Includes presentKHR, which can block on the display
        std::vector<float> fenceWaitMs;
        bool screenshotSaved = false;
        std::optional<Renderer::TraversalCounts> traversal;
        uint64_t pixels = 0;
        std::vector<Renderer::TraceOnlyQueueResult> traceOnly;
    };

    BenchmarkSpecification m_Spec;
    std::vector<BenchmarkView> m_Views;
    std::vector<ViewSamples> m_Samples;
    size_t m_ViewIndex = 0;
    uint32_t m_FramesInView = 0;
    uint32_t m_CountingFramesLeft = 0;
    bool m_Capturing = false;
};

} // namespace Momo
