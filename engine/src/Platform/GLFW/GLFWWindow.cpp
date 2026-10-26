#include "Momo/Window.h"
#include <cstdint>
#include <stdexcept>
#include <string>
#include "Momo/Input/InputState.h"
#include "Momo/Logging/Logger.h"

#include <GLFW/glfw3.h>
namespace Momo
{
    class GLFWWindow : public IWindow {
    public:
        explicit GLFWWindow(const WindowSpecification& spec) {
            if (s_InitCount == 0) {
                if (!glfwInit())
                    throw std::runtime_error("glfwInit failed");
                // (Optional) glfwSetErrorCallback([](int code, const char* desc){ std::cerr << "GLFW " << code << ": " << desc << "\n"; });
            }
            ++s_InitCount;

            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // we’ll use Vulkan later
            m_Window = glfwCreateWindow(static_cast<int>(spec.Width),
                                        static_cast<int>(spec.Height),
                                        spec.Title.c_str(),
                                        nullptr, nullptr);
            if (!m_Window) {
                --s_InitCount;
                if (s_InitCount == 0) glfwTerminate();
                throw std::runtime_error("glfwCreateWindow failed");
            }

            m_Width  = spec.Width;
            m_Height = spec.Height;

            // Hook resize if you want
            glfwSetWindowUserPointer(m_Window, this);
            glfwSetInputMode(m_Window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            glfwSetFramebufferSizeCallback(m_Window, [](GLFWwindow* win, int w, int h){
                auto* self = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(win));
                self->m_Width  = static_cast<std::uint32_t>(w);
                self->m_Height = static_cast<std::uint32_t>(h);
            });
        }

        ~GLFWWindow() override {
            if (m_Window) {
                glfwDestroyWindow(m_Window);
                m_Window = nullptr;
            }
            --s_InitCount;
            if (s_InitCount == 0)
                glfwTerminate();
        }

        Input::InputState ReadInput() override {
            Input::InputState inputState{};
            if (glfwGetKey(m_Window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                // glfwSetWindowShouldClose(m_Window, true);
                inputState.keys.SetKeyState(Input::Key::Escape, true);
                LOG_INFO("Input", "Escape key pressed");
            }
            if (glfwGetKey(m_Window, GLFW_KEY_W) == GLFW_PRESS) {
                inputState.keys.SetKeyState(Input::Key::W, true);
            }
            if (glfwGetKey(m_Window, GLFW_KEY_A) == GLFW_PRESS) {
                inputState.keys.SetKeyState(Input::Key::A, true);
            }
            if (glfwGetKey(m_Window, GLFW_KEY_S) == GLFW_PRESS) {
                inputState.keys.SetKeyState(Input::Key::S, true);
            }
            if (glfwGetKey(m_Window, GLFW_KEY_D) == GLFW_PRESS) {
                inputState.keys.SetKeyState(Input::Key::D, true);
            }
            if (glfwGetKey(m_Window, GLFW_KEY_Q) == GLFW_PRESS) {
                inputState.keys.SetKeyState(Input::Key::Q, true);
            }
            if (glfwGetKey(m_Window, GLFW_KEY_E) == GLFW_PRESS) {
                inputState.keys.SetKeyState(Input::Key::E, true);
            }
            if (glfwGetKey(m_Window, GLFW_KEY_SPACE) == GLFW_PRESS) {
                inputState.keys.SetKeyState(Input::Key::Space, true);
            }
            if (glfwGetKey(m_Window, GLFW_KEY_P) == GLFW_PRESS) {
                inputState.keys.SetKeyState(Input::Key::P, true);
            }
            glfwGetCursorPos(m_Window, &inputState.mouseX, &inputState.mouseY);
            inputState.windowFocused = glfwGetWindowAttrib(m_Window, GLFW_FOCUSED) == GLFW_TRUE;

            // LOG_INFO("Input", "Mouse Position: ({}, {})", inputState.mouseX, inputState.mouseY);
            return inputState;
        }

        void PollEvents() override { glfwPollEvents(); }
        bool ShouldClose() const override { return glfwWindowShouldClose(m_Window); }

        void* GetNativeHandle() const override { return m_Window; }
        std::uint32_t GetWidth()  const override { return m_Width; }
        std::uint32_t GetHeight() const override { return m_Height; }

    private:
        GLFWwindow* m_Window = nullptr;
        std::uint32_t m_Width = 0, m_Height = 0;
        Input::Key translateGLFWKey(int glfwKey) {
            switch (glfwKey) {
                case GLFW_KEY_W: return Input::Key::W;
                case GLFW_KEY_A: return Input::Key::A;
                case GLFW_KEY_S: return Input::Key::S;
                case GLFW_KEY_D: return Input::Key::D;
                case GLFW_KEY_Q: return Input::Key::Q;
                case GLFW_KEY_E: return Input::Key::E;
                case GLFW_KEY_SPACE: return Input::Key::Space;
                case GLFW_KEY_ESCAPE: return Input::Key::Escape;
                case GLFW_KEY_P: return Input::Key::P;
                default: return Input::Key::Escape; // Default fallback
            }
        }

        static inline int s_InitCount = 0;
    };

    // Factory
    IWindow* IWindow::Create(const WindowSpecification& spec) {
        return new GLFWWindow(spec);
    }
}
