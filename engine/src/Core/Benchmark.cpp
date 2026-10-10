#include "Momo/Core/Benchmark.h"
#include "Momo/Logging/Logger.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <format>
#include <fstream>
#include <numeric>

namespace Momo {
namespace {

// Camera views of the Sponza scene (world units after its 0.008 scale; the
// long courtyard runs along x). Chosen to cover different ray behaviour:
// mostly direct light, deep indirect light, and rays escaping to the sky.
std::vector<BenchmarkView> SponzaViews()
{
    return {
        { "Courtyard", glm::vec3(-11.0f, 2.0f, 0.0f), 0.0f, 5.0f },
        { "Gallery", glm::vec3(9.0f, 1.8f, -5.0f), 180.0f, 0.0f },
        { "Sky", glm::vec3(0.0f, 1.5f, 0.0f), 0.0f, 85.0f },
    };
}

std::string StatsJson(std::vector<float> values)
{
    std::sort(values.begin(), values.end());
    double average = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    return std::format(R"({{ "avg": {:.2f}, "median": {:.2f}, "min": {:.2f}, "max": {:.2f} }})",
                       average, values[values.size() / 2], values.front(), values.back());
}

// JSON string escaping for the few characters a GPU name could contain
std::string JsonString(const std::string& text)
{
    std::string escaped = "\"";
    for (char c : text)
    {
        if (c == '"' || c == '\\')
            escaped += '\\';
        escaped += c;
    }
    return escaped + "\"";
}

} // namespace

Benchmark::Benchmark(BenchmarkSpecification spec)
    : m_Spec(std::move(spec)), m_Views(SponzaViews()), m_Samples(m_Views.size())
{
    LOG_INFO("Benchmark", "Benchmarking {} views: {} warmup + {} measured frames each", m_Views.size(), m_Spec.WarmupFrames, m_Spec.MeasuredFrames);
}

std::optional<size_t> Benchmark::Record(const Renderer::DrawFrameTime& drawTime, const Profiling::FrameTime& frameTime)
{
    if (IsFinished())
        return std::nullopt;

    if (m_CountingFramesLeft > 0)
    {
        // After the last counting frame, one capture frame
        if (--m_CountingFramesLeft == 0)
            m_Capturing = true;
        return std::nullopt;
    }
    if (m_Capturing)
    {
        m_Capturing = false;
        LOG_INFO("Benchmark", "View '{}' done", m_Views[m_ViewIndex].Name);
        m_FramesInView = 0;
        return m_ViewIndex++;
    }

    m_FramesInView++;
    // Timings describe a frame from MAX_FRAMES_IN_FLIGHT frames ago; the warmup
    // makes sure that frame already showed this view
    if (m_FramesInView > m_Spec.WarmupFrames && drawTime.hasRayTracingMs)
    {
        ViewSamples& samples = m_Samples[m_ViewIndex];
        samples.dispatchMs.push_back(drawTime.rayTracingMs);
        samples.frameMs.push_back(drawTime.gpuFrameMs);
        samples.cpuTotalMs.push_back(frameTime.cpuTotalMs);
        samples.recordMs.push_back(frameTime.recordMs);
        samples.submitMs.push_back(frameTime.submitMs);
        samples.fenceWaitMs.push_back(frameTime.fenceWaitMs);
    }

    // Timing done: render the counting frames next
    if (m_Samples[m_ViewIndex].dispatchMs.size() == m_Spec.MeasuredFrames)
        m_CountingFramesLeft = std::max(m_Spec.CountingFrames, 1u);
    return std::nullopt;
}

void Benchmark::SetTraversalCounts(size_t viewIndex, const Renderer::TraversalCounts& counts, uint64_t pixels)
{
    m_Samples[viewIndex].traversal = counts;
    m_Samples[viewIndex].pixels = pixels * std::max(m_Spec.CountingFrames, 1u);
}

std::filesystem::path Benchmark::ScreenshotPath(size_t viewIndex) const
{
    std::string fileName = m_Views[viewIndex].Name;
    std::transform(fileName.begin(), fileName.end(), fileName.begin(), [](unsigned char c) { return std::tolower(c); });
    return m_Spec.OutputPath.parent_path() / (fileName + ".jpg");
}

bool Benchmark::WriteResults(const BenchmarkEnvironment& environment) const
{
    auto today = std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now());

    std::string views;
    for (size_t i = 0; i < m_Views.size() && i < m_ViewIndex; i++)
    {
        const BenchmarkView& view = m_Views[i];
        std::string traversal;
        if (const auto& counts = m_Samples[i].traversal)
        {
            using namespace Renderer;
            const uint64_t* c = counts->values;
            double rays = double(c[ClosestRays] + c[ShadowRays]);
            double pixels = double(m_Samples[i].pixels);
            traversal = std::format(R"(,
      "traversal": {{ "closest_rays_per_pixel": {:.3f}, "shadow_rays_per_pixel": {:.3f}, "steps_per_ray": {:.2f}, "node_reads_per_ray": {:.2f}, "node_bytes_per_ray": {:.1f}, "box_tests_per_ray": {:.2f}, "triangle_tests_per_ray": {:.2f}, "alpha_tests_per_ray": {:.3f} }})",
                c[ClosestRays] / pixels, c[ShadowRays] / pixels, c[Steps] / rays, c[NodeReads] / rays,
                c[NodeReads] * double(environment.NodeBytes) / rays, c[BoxTests] / rays, c[TriangleTests] / rays, c[AlphaTests] / rays);
        }
        std::string traceOnly;
        if (!m_Samples[i].traceOnly.empty())
        {
            // The same rays the path tracer traced in one frame, traced again by
            // the trace-only kernel; compared with the path tracer's own rate
            uint64_t closestRays = 0, shadowRays = 0;
            double closestMs = 0.0, shadowMs = 0.0;
            std::string queues;
            for (const Renderer::TraceOnlyQueueResult& queue : m_Samples[i].traceOnly)
            {
                (queue.shadow ? shadowRays : closestRays) += queue.rays;
                (queue.shadow ? shadowMs : closestMs) += queue.medianMs;
                queues += std::format(R"({}{{ "kind": "{}", "bounce": {}, "rays": {}, "ms": {:.3f} }})",
                    queues.empty() ? "" : ", ", queue.shadow ? "shadow" : "closest", queue.bounce, queue.rays, queue.medianMs);
            }
            auto mraysPerSecond = [](double rays, double ms) { return ms > 0.0 ? rays / ms / 1000.0 : 0.0; };
            std::string megakernel = "null";
            if (const auto& counts = m_Samples[i].traversal; counts && !m_Samples[i].dispatchMs.empty())
            {
                std::vector<float> sorted = m_Samples[i].dispatchMs;
                std::sort(sorted.begin(), sorted.end());
                double raysPerFrame = double(counts->values[Renderer::ClosestRays] + counts->values[Renderer::ShadowRays])
                                    / std::max(m_Spec.CountingFrames, 1u);
                megakernel = std::format("{:.1f}", mraysPerSecond(raysPerFrame, sorted[sorted.size() / 2]));
            }
            traceOnly = std::format(R"(,
      "trace_only": {{ "closest_rays": {}, "shadow_rays": {}, "closest_ms": {:.2f}, "shadow_ms": {:.2f}, "mrays_per_s": {:.1f}, "closest_mrays_per_s": {:.1f}, "shadow_mrays_per_s": {:.1f}, "megakernel_mrays_per_s": {},
        "queues": [{}] }})",
                closestRays, shadowRays, closestMs, shadowMs, mraysPerSecond(double(closestRays + shadowRays), closestMs + shadowMs),
                mraysPerSecond(double(closestRays), closestMs), mraysPerSecond(double(shadowRays), shadowMs), megakernel, queues);
        }
        std::string screenshot = m_Samples[i].screenshotSaved
            ? std::format(R"(,
      "screenshot": {})", JsonString(ScreenshotPath(i).filename().string()))
            : "";
        views += std::format(R"({}    {{ "name": {}, "position": [{}, {}, {}], "yaw": {}, "pitch": {},
      "dispatch_ms": {},
      "frame_ms": {},
      "cpu_total_ms": {},
      "record_ms": {},
      "submit_ms": {},
      "fence_wait_ms": {}{}{}{} }})",
            i > 0 ? ",\n" : "", JsonString(view.Name), view.Position.x, view.Position.y, view.Position.z, view.Yaw, view.Pitch,
            StatsJson(m_Samples[i].dispatchMs), StatsJson(m_Samples[i].frameMs), StatsJson(m_Samples[i].cpuTotalMs),
            StatsJson(m_Samples[i].recordMs), StatsJson(m_Samples[i].submitMs), StatsJson(m_Samples[i].fenceWaitMs), traversal, traceOnly, screenshot);
    }

    std::string json = std::format(R"({{
  "date": "{:%Y-%m-%d}",
  "gpu": {},
  "resolution": "{}x{}",
  "bounces": {},
  "warmup_frames": {},
  "frames_per_view": {},
  "bvh": {{ "layout": "compressed 8-wide", "triangles": {}, "nodes": {}, "max_depth": {}, "binary_nodes": {}, "leaves": {}, "binary_max_depth": {}, "max_leaf_triangles": {}, "sah_cost": {:.2f}, "build_ms": {:.1f}, "gpu_mb": {:.1f} }},
  "views": [
{}
  ]
}}
)", today, JsonString(environment.Gpu), environment.Width, environment.Height, environment.Bounces,
        m_Spec.WarmupFrames, m_Spec.MeasuredFrames,
        environment.Triangles, environment.Bvh.nodeCount, environment.Bvh.maxDepth, environment.Bvh.binary.nodeCount,
        environment.Bvh.binary.leafCount, environment.Bvh.binary.maxDepth, environment.Bvh.binary.maxLeafTriangles,
        environment.Bvh.binary.sahCost, environment.Bvh.buildMs, environment.GpuBytes / (1024.0 * 1024.0),
        views);

    std::ofstream file(m_Spec.OutputPath);
    if (!file)
    {
        LOG_ERROR("Benchmark", "Couldn't write results to {}", m_Spec.OutputPath.string());
        return false;
    }
    file << json;
    LOG_INFO("Benchmark", "Results written to {}", m_Spec.OutputPath.string());
    return true;
}

} // namespace Momo
