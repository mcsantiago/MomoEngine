#include "Momo/Renderer/Camera.h"
#include "Momo/Logging/Logger.h"
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>

namespace Momo {
Camera::Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch)
    : m_Position(position), m_WorldUp(up), m_Yaw(yaw), m_Pitch(pitch), m_Front(glm::vec3(0.0f, 0.0f, -1.0f)), m_FirstMouse(true) {
    UpdateCameraVectors(0, 0);
}

Camera Camera::LookingAlong(glm::vec3 position, glm::vec3 up, glm::vec3 forward) {
    // Inverts UpdateCameraVectors: front = (cos(yaw) cos(pitch), sin(pitch), sin(yaw) cos(pitch))
    float pitch = glm::degrees(std::asin(glm::clamp(forward.y, -1.0f, 1.0f)));
    float yaw = glm::degrees(std::atan2(forward.z, forward.x));
    return Camera(position, up, yaw, pitch);
}

void Camera::OnUpdate(float deltaTime, const Input::InputState& inputState) {
    if (m_FirstMouse) {
        m_LastMouseX = inputState.mouseX;
        m_LastMouseY = inputState.mouseY;
        m_FirstMouse = false;
    }

    // The cursor position keeps changing while other windows have focus; only
    // turn the camera for movement made while ours does. Tracking the position
    // regardless keeps the camera from jumping when focus comes back.
    int deltaX = inputState.windowFocused ? inputState.mouseX - m_LastMouseX : 0;
    int deltaY = inputState.windowFocused ? inputState.mouseY - m_LastMouseY : 0;
    m_LastMouseX = inputState.mouseX;
    m_LastMouseY = inputState.mouseY;

    m_Position += inputState.keys.IsKeyPressed(Input::Key::W) ? m_Front * static_cast<float>(deltaTime) * m_MovementSpeed : glm::vec3(0.0f);
    m_Position += inputState.keys.IsKeyPressed(Input::Key::S) ? -m_Front * static_cast<float>(deltaTime) * m_MovementSpeed : glm::vec3(0.0f);
    m_Position += inputState.keys.IsKeyPressed(Input::Key::A) ? -m_Right * static_cast<float>(deltaTime) * m_MovementSpeed : glm::vec3(0.0f);
    m_Position += inputState.keys.IsKeyPressed(Input::Key::D) ? m_Right * static_cast<float>(deltaTime) * m_MovementSpeed : glm::vec3(0.0f);
    // m_Position += inputState.keys.IsKeyPressed(Input::Key::Q) ? -m_Up * static_cast<float>(deltaTime) : glm::vec3(0.0f);
    // m_Position += inputState.keys.IsKeyPressed(Input::Key::E) ? m_Up * static_cast<float>(deltaTime) : glm::vec3(0.0f);

    UpdateCameraVectors(deltaX, deltaY);
}

void Camera::SetPose(glm::vec3 position, float yaw, float pitch) {
    m_Position = position;
    m_Yaw = yaw;
    m_Pitch = pitch;
    UpdateCameraVectors(0, 0);
}

glm::mat4 Camera::GetViewMatrix() const {
    return glm::lookAt(m_Position, m_Position + m_Front, m_Up);
}

void Camera::UpdateCameraVectors(int deltaX, int deltaY) {
    m_Yaw += deltaX * m_MouseSensitivity;
    m_Pitch -= deltaY * m_MouseSensitivity;

    if (m_Pitch > 89.0f)
        m_Pitch = 89.0f;
    if (m_Pitch < -89.0f)
        m_Pitch = -89.0f;

    glm::vec3 front;
    front.x = cos(glm::radians(m_Yaw)) * cos(glm::radians(m_Pitch));
    front.y = sin(glm::radians(m_Pitch));
    front.z = sin(glm::radians(m_Yaw)) * cos(glm::radians(m_Pitch));
    m_Front = glm::normalize(front);
    m_Right = glm::normalize(glm::cross(m_Front, m_WorldUp));
    m_Up = glm::normalize(glm::cross(m_Right, m_Front));

    LOG_DEBUG("Camera", "Camera Vectors Updated: Front({}, {}, {}), Right({}, {}, {}), Up({}, {}, {}), Delta({}, {})",
              m_Front.x, m_Front.y, m_Front.z,
              m_Right.x, m_Right.y, m_Right.z,
              m_Up.x, m_Up.y, m_Up.z, deltaX, deltaY);
}
} // namespace Momo
