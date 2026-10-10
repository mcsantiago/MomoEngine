#include "Momo/Core/Benchmark.h"
#include "Momo/Logging/Logger.h"

#include <algorithm>
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

void Benchmark::Record(const Renderer::DrawFrameTime& frameTime)
{
    if (IsFinished())
        return;

    m_FramesInView++;
    // Timings describe a frame from MAX_FRAMES_IN_FLIGHT frames ago; the warmup
    // makes sure that frame already showed this view
    if (m_FramesInView > m_Spec.WarmupFrames && frameTime.hasRayTracingMs)
    {
        m_Samples[m_ViewIndex].dispatchMs.push_back(frameTime.rayTracingMs);
        m_Samples[m_ViewIndex].frameMs.push_back(frameTime.gpuFrameMs);
    }

    if (m_Samples[m_ViewIndex].dispatchMs.size() == m_Spec.MeasuredFrames)
    {
        LOG_INFO("Benchmark", "View '{}' done", m_Views[m_ViewIndex].Name);
        m_ViewIndex++;
        m_FramesInView = 0;
    }
}

bool Benchmark::WriteResults(const BenchmarkEnvironment& environment) const
{
    auto today = std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now());

    std::string views;
    for (size_t i = 0; i < m_Views.size() && i < m_ViewIndex; i++)
    {
        const BenchmarkView& view = m_Views[i];
        views += std::format(R"({}    {{ "name": {}, "position": [{}, {}, {}], "yaw": {}, "pitch": {},
      "dispatch_ms": {},
      "frame_ms": {} }})",
            i > 0 ? ",\n" : "", JsonString(view.Name), view.Position.x, view.Position.y, view.Position.z, view.Yaw, view.Pitch,
            StatsJson(m_Samples[i].dispatchMs), StatsJson(m_Samples[i].frameMs));
    }

    std::string json = std::format(R"({{
  "date": "{:%Y-%m-%d}",
  "gpu": {},
  "resolution": "{}x{}",
  "bounces": {},
  "warmup_frames": {},
  "frames_per_view": {},
  "views": [
{}
  ]
}}
)", today, JsonString(environment.Gpu), environment.Width, environment.Height, environment.Bounces,
        m_Spec.WarmupFrames, m_Spec.MeasuredFrames, views);

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
