#pragma once

#include <glm/glm.hpp>
#include "Momo/Input/InputState.h"

namespace Momo {
class Camera {
public:
    Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch);

    glm::mat4 GetViewMatrix() const;
    void OnUpdate(float deltaTime, const Input::InputState& inputState);

private:
    void UpdateCameraVectors(int deltaX, int deltaY);

    glm::vec3 m_Position;
    glm::vec3 m_Front;
    glm::vec3 m_Up;
    glm::vec3 m_Right;
    glm::vec3 m_WorldUp;

    float m_Yaw;
    float m_Pitch;
    float m_MovementSpeed = 2.5f;

    bool m_FirstMouse = true;
    double m_LastMouseX = 0;
    double m_LastMouseY = 0;
    double m_MouseSensitivity = 0.1f;
};
} // namespace Momo
