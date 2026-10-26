#pragma once

#include <glm/glm.hpp>
#include "Momo/Input/InputState.h"

namespace Momo {
class Camera {
public:
    Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch);
    // Looking along forward (unit length, not straight up or down)
    static Camera LookingAlong(glm::vec3 position, glm::vec3 up, glm::vec3 forward);

    glm::mat4 GetViewMatrix() const;
    void OnUpdate(float deltaTime, const Input::InputState& inputState);
    // Places the camera directly (yaw and pitch in degrees), e.g. for benchmark views
    void SetPose(glm::vec3 position, float yaw, float pitch);

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
