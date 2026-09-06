#!/bin/bash
set -e

# 1. BALL_CLOSE_AREA_PX: 65000 -> 180000
sed -i 's#const double BALL_CLOSE_AREA_PX = 65000.0;#const double BALL_CLOSE_AREA_PX = 180000.0;#' main_fetch.cpp
grep -q "BALL_CLOSE_AREA_PX = 180000.0;" main_fetch.cpp || { echo "FAILED: BALL_CLOSE_AREA_PX"; exit 1; }

# 2. SIDESTEP: speed 0.15 -> 0.4, steps 4 -> 5
sed -i 's#const float SIDESTEP_SPEED = 0.15f;#const float SIDESTEP_SPEED = 0.4f;#' main_fetch.cpp
sed -i 's#const int SIDESTEP_STEPS = 4;#const int SIDESTEP_STEPS = 5;#' main_fetch.cpp
grep -q "SIDESTEP_SPEED = 0.4f;" main_fetch.cpp || { echo "FAILED: SIDESTEP_SPEED"; exit 1; }
grep -q "SIDESTEP_STEPS = 5;" main_fetch.cpp || { echo "FAILED: SIDESTEP_STEPS"; exit 1; }

# 3. WALK_PAST: speed 0.3 -> 0.53, steps 6 -> 15
sed -i 's#const float WALK_PAST_SPEED = 0.3f;#const float WALK_PAST_SPEED = 0.53f;#' main_fetch.cpp
sed -i 's#const int WALK_PAST_STEPS = 6;#const int WALK_PAST_STEPS = 15;#' main_fetch.cpp
grep -q "WALK_PAST_SPEED = 0.53f;" main_fetch.cpp || { echo "FAILED: WALK_PAST_SPEED"; exit 1; }
grep -q "WALK_PAST_STEPS = 15;" main_fetch.cpp || { echo "FAILED: WALK_PAST_STEPS"; exit 1; }

# 4. settle_pause() -- only insert if it isn't already there
if grep -q "void settle_pause()" main_fetch.cpp; then
    echo "settle_pause() already present, skipping insertion"
else
    sed -i '\#BlobResult find_vest(const cv::Mat& frame, cv::aruco::ArucoDetector& detector)#i\
// SETTLE / STOP-BETWEEN-PHASES: brief full stop between blind sub-phases so each maneuver has a\
// clean, distinct start instead of one command blending directly into the next.\
const float SETTLE_PAUSE_SEC = 0.5f; // NEEDS CALIBRATION\
void settle_pause()\
{\
    isaaclab::autopilot::set(0.0f, 0.0f, 0.0f);\
    std::cout << "[fetch] settling...\\n";\
    std::this_thread::sleep_for(std::chrono::milliseconds((int)(SETTLE_PAUSE_SEC * 1000)));\
}' main_fetch.cpp
    grep -q "void settle_pause()" main_fetch.cpp || { echo "FAILED: settle_pause definition"; exit 1; }
fi

# 5. settle_pause() calls at each transition -- only insert if not already there
if grep -q "settle_pause();" main_fetch.cpp; then
    echo "settle_pause() calls already present, skipping insertion"
else
    sed -i '/close to ball -> SIDESTEP/a\                    settle_pause();' main_fetch.cpp
    sed -i '/sidestep complete -> WALK_PAST/a\                settle_pause();' main_fetch.cpp
    sed -i '/walk past complete -> TURN_AROUND/a\                settle_pause();' main_fetch.cpp
    sed -i '/turn around complete -> SIDESTEP_RECENTER/a\                settle_pause();' main_fetch.cpp
    sed -i '/recenter complete -> ALIGN/a\                settle_pause();' main_fetch.cpp
    count=$(grep -c "settle_pause();" main_fetch.cpp)
    [ "$count" -eq 5 ] || { echo "FAILED: expected 5 settle_pause() calls, found $count"; exit 1; }
fi

echo "ALL EDITS APPLIED SUCCESSFULLY."
