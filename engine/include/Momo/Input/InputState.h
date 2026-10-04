#pragma once
#include <glm/glm.hpp>
#include <vector>
#include "Key.h"


namespace Momo
{
namespace Input {

enum class Action {
    MoveForward,
    MoveBackward,
    MoveLeft,
    MoveRight,
    MoveUp,
    MoveDown,
    Jump,
    UIAccept,
    UIQuit
};


class KeyStates {
private:
    bool m_Keys[KeyCount] = { false };
public:
    bool IsKeyPressed(Key key) const {
        return m_Keys[static_cast<size_t>(key)];
    }
    void SetKeyState(Key key, bool pressed) {
        m_Keys[static_cast<size_t>(key)] = pressed;
    }
};

struct Binding {
    enum class Source {
        Key,
        MouseButton,
        GamepadButton,
        GamepadAxis
    } source;

    int code;           // Key::W, GLFW_GAMEPAD_BUTTON_A, etc.
    float scale = 1.0f; // -1 to bind the negative half of an axis
};

struct InputState {
    KeyStates keys;
    bool mouseButtons[8] = { false };
    double mouseX = 0;
    double mouseY = 0;
};

class InputManager {
public:
    void BindAction(Action action, const Binding& binding) {
        m_Bindings.push_back({action, binding});
    }

    bool IsActionPressed(Action action, const InputState& state) const {
        for (const auto& binding : m_Bindings) {
            if (binding.action == action) {
                if (binding.binding.source == Binding::Source::Key) {
                    if (state.keys.IsKeyPressed(static_cast<Key>(binding.binding.code))) {
                        return true;
                    }
                }
                else if (binding.binding.source == Binding::Source::MouseButton) {
                    if (state.mouseButtons[binding.binding.code]) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

private:
    struct ActionBinding {
        Action action;
        Binding binding;
    };
    std::vector<ActionBinding> m_Bindings;
};

} // namespace Input
} // namespace Momo
