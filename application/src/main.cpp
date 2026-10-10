#include <Momo/Core/Application.h>
#include <Momo/Logging/Logger.h>

class AppLayer : public Momo::Layer
{
public:
    void OnAttach() override
    {
        LOG_INFO("AppLayer", "Application layer attached.");
    }

    void OnUpdate(float dt) override
    {
        // Update logic here
    }
};

bool ParseCommandLineArguments(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i)
    {
        // --log-level=<level>
        if (strncmp(argv[i], "--log-level=", 12) == 0)
        {
            const char* levelStr = argv[i] + 12;
            if (strcmp(levelStr, "trace") == 0) {
                Momo::Logger::SetLevel(Momo::LogLevel::Trace);
                return true;
            } else if (strcmp(levelStr, "info") == 0) {
                Momo::Logger::SetLevel(Momo::LogLevel::Info);
                return true;
            } else if (strcmp(levelStr, "debug") == 0) {
                Momo::Logger::SetLevel(Momo::LogLevel::Debug);
                return true;
            } else if (strcmp(levelStr, "warn") == 0) {
                Momo::Logger::SetLevel(Momo::LogLevel::Warning);
                return true;
            } else if (strcmp(levelStr, "error") == 0) {
                Momo::Logger::SetLevel(Momo::LogLevel::Error);
                return true;
            } else if (strcmp(levelStr, "fatal") == 0) {
                Momo::Logger::SetLevel(Momo::LogLevel::Fatal);
                return true;
            }
        }
    }
    return false;
}

void SetLogLevelFromEnvironment()
{
    if (std::getenv("MOMO_LOG_LEVEL"))
    {
        const char* levelStr = std::getenv("MOMO_LOG_LEVEL");
        if (strcmp(levelStr, "trace") == 0) {
            Momo::Logger::SetLevel(Momo::LogLevel::Trace);
        } else if (strcmp(levelStr, "debug") == 0) {
            Momo::Logger::SetLevel(Momo::LogLevel::Debug);
        } else if (strcmp(levelStr, "info") == 0) {
            Momo::Logger::SetLevel(Momo::LogLevel::Info);
        } else if (strcmp(levelStr, "warn") == 0) {
            Momo::Logger::SetLevel(Momo::LogLevel::Warning);
        } else if (strcmp(levelStr, "error") == 0) {
            Momo::Logger::SetLevel(Momo::LogLevel::Error);
        } else if (strcmp(levelStr, "fatal") == 0) {
            Momo::Logger::SetLevel(Momo::LogLevel::Fatal);
        }
    }
}

// MOMO_RENDER_MODE=raytraced starts in ray traced mode (P toggles at runtime)
Momo::Renderer::RenderMode GetRenderModeFromEnvironment()
{
    const char* modeStr = std::getenv("MOMO_RENDER_MODE");
    if (modeStr && strcmp(modeStr, "raytraced") == 0) {
        return Momo::Renderer::RenderMode::RayTraced;
    }
    return Momo::Renderer::RenderMode::Raster;
}

// MOMO_BENCHMARK=<results.json> runs the ray tracing benchmark and exits
std::optional<Momo::BenchmarkSpecification> GetBenchmarkFromEnvironment()
{
    const char* outputPath = std::getenv("MOMO_BENCHMARK");
    if (!outputPath || outputPath[0] == '\0') {
        return std::nullopt;
    }
    return Momo::BenchmarkSpecification{ .OutputPath = outputPath };
}

int main(int argc, char** argv)
{
    // Read command-line arguments (if any)
    bool logLevelSet = ParseCommandLineArguments(argc, argv);
    if (!logLevelSet)
    {
        // If no log level was specified via command-line arguments, fall back to environment variable
        SetLogLevelFromEnvironment();
    }

    Momo::ApplicationSpecification appSpec;
    appSpec.Name = "Hyperbolic Time Chamber v0.1";
    appSpec.WindowSpec.Width = 1920;
    appSpec.WindowSpec.Height = 1080;
    appSpec.InitialRenderMode = GetRenderModeFromEnvironment();
    appSpec.Benchmark = GetBenchmarkFromEnvironment();

    Momo::Application app(appSpec);
    app.PushLayer<AppLayer>();
    app.Run();

    // Explicitly shutdown before destruction
    // This ensures Vulkan instance is destroyed before GLFW terminates
    app.Shutdown();

    return 0;
}