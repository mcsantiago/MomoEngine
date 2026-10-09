#pragma once

#include <chrono>

namespace Momo::Profiling
{
class Timer
{
public:
    Timer() = default;
    ~Timer() = default;

    void Start()
    {
        m_startTime = std::chrono::steady_clock::now();
    }

    void End()
    {
        m_endTime = std::chrono::steady_clock::now();
    }

    double GetElapsedTimeInMs() const
    {
        return std::chrono::duration<double, std::milli>(m_endTime - m_startTime).count();
    }
private:
    std::chrono::steady_clock::time_point m_startTime;
    std::chrono::steady_clock::time_point m_endTime;
};
} // namespace Momo::Profiling
