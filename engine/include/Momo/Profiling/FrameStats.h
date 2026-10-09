#pragma once

#include <algorithm>
#include <cstdint>
#include <format>
#include <limits>
#include <string>
#include <Momo/Logging/Logger.h>

namespace Momo::Profiling
{
struct FrameTime
{
    uint64_t frame;
    float cpuTotalMs, updateMs, recordMs, submitMs, fenceWaitMs, gpuMs;
};

// Running avg/min/max of a single timing over the current window
struct StatWindow
{
    double sum = 0.0;
    float min = std::numeric_limits<float>::max();
    float max = 0.0f;

    void Add(float value)
    {
        sum += value;
        min = std::min(min, value);
        max = std::max(max, value);
    }

    // "avg (min-max)"
    std::string Format(uint32_t count) const
    {
        if (count == 0)
            return "n/a";
        return std::format("{:.2f} ({:.2f}-{:.2f})", sum / count, min, max);
    }
};

// Collects per-frame timings and logs one summary per window
class FrameStatsAccumulator
{
public:
    void Add(const FrameTime& frameTime)
    {
        m_CpuTotal.Add(frameTime.cpuTotalMs);
        m_Update.Add(frameTime.updateMs);
        m_Record.Add(frameTime.recordMs);
        m_Submit.Add(frameTime.submitMs);
        m_FenceWait.Add(frameTime.fenceWaitMs);
        m_Gpu.Add(frameTime.gpuMs);
        m_LastFrame = frameTime.frame;
        m_Count++;
    }

    void LogAndReset(double windowSeconds)
    {
        double fps = windowSeconds > 0.0 ? m_Count / windowSeconds : 0.0;
        LOG_INFO("Momo", "Frame {} | {:.0f} fps | ms avg (min-max) | CPU total: {}, Update: {}, Record: {}, Submit: {}, Fence Wait: {}, GPU: {}",
            m_LastFrame, fps,
            m_CpuTotal.Format(m_Count), m_Update.Format(m_Count), m_Record.Format(m_Count),
            m_Submit.Format(m_Count), m_FenceWait.Format(m_Count), m_Gpu.Format(m_Count));
        *this = FrameStatsAccumulator{};
    }

private:
    uint64_t m_LastFrame = 0;
    uint32_t m_Count = 0;
    StatWindow m_CpuTotal, m_Update, m_Record, m_Submit, m_FenceWait, m_Gpu;
};
} // namespace Momo::Profiling
