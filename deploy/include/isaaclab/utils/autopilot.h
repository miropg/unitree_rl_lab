// Autopilot override: lets code (not a joystick) drive the velocity_commands
// observation. When active, velocity_commands ignores the joystick and uses
// these values instead. Real remote-control behavior is unchanged when
// autopilot is not active.
#pragma once
#include <atomic>

namespace isaaclab
{
namespace autopilot
{

inline std::atomic<bool> active{false};
inline std::atomic<float> lin_vel_x{0.0f};
inline std::atomic<float> lin_vel_y{0.0f};
inline std::atomic<float> ang_vel_z{0.0f};

inline void set(float vx, float vy, float wz)
{
    lin_vel_x.store(vx);
    lin_vel_y.store(vy);
    ang_vel_z.store(wz);
    active.store(true);
}

inline void stop()
{
    lin_vel_x.store(0.0f);
    lin_vel_y.store(0.0f);
    ang_vel_z.store(0.0f);
    // leave active=true so it holds a zero command instead of reverting to joystick
}

inline void release()
{
    active.store(false);
}

} // namespace autopilot
} // namespace isaaclab
