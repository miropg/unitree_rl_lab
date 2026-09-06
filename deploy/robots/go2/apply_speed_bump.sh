#!/bin/bash
set -e

# 1. Flatten APPROACH_BALL to one constant speed -- decel math untouched, both endpoints now equal
sed -i 's#MAX_FORWARD_SPEED = 0.4f;#MAX_FORWARD_SPEED = 0.5f;#' main_fetch.cpp
sed -i 's#MIN_APPROACH_SPEED = 0.20f;#MIN_APPROACH_SPEED = 0.5f;#' main_fetch.cpp
sed -i "s#// don't fully stop while still approaching, just slow down#// flat approach speed (no deceleration) -- equal to MAX_FORWARD_SPEED; NEEDS RECHECK if limping persists under autopilot#" main_fetch.cpp
grep -q "MAX_FORWARD_SPEED = 0.5f;" main_fetch.cpp || { echo "FAILED: MAX_FORWARD_SPEED"; exit 1; }
grep -q "MIN_APPROACH_SPEED = 0.5f;" main_fetch.cpp || { echo "FAILED: MIN_APPROACH_SPEED"; exit 1; }

# 2. WALK_PAST_SPEED bump: 0.53 -> 0.6
sed -i 's#WALK_PAST_SPEED = 0.53f;#WALK_PAST_SPEED = 0.6f;#' main_fetch.cpp
grep -q "WALK_PAST_SPEED = 0.6f;" main_fetch.cpp || { echo "FAILED: WALK_PAST_SPEED"; exit 1; }

echo "ALL EDITS APPLIED SUCCESSFULLY."
