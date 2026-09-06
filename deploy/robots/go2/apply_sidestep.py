import sys

path = "main_fetch.cpp"
src = open(path).read()

edits = []

edits.append((
'''                if (approach_close_confirm.confirm(ball.area >= BALL_CLOSE_AREA_PX, APPROACH_CLOSE_CONFIRM_FRAMES)) {
                    // Switch to ALIGN/ORBITINg
                    phase = Phase::ALIGN;
                    // Print message confirming switch to ALIGN
                    std::cout << "[fetch] close to ball -> ALIGN\\n";
                }''',
'''                if (approach_close_confirm.confirm(ball.area >= BALL_CLOSE_AREA_PX, APPROACH_CLOSE_CONFIRM_FRAMES)) {
                    phase = Phase::SIDESTEP;
                    sidestep_counter = 0;
                    std::cout << "[fetch] close to ball -> SIDESTEP\\n";
                }'''
))

edits.append((
'''        // Only run if currently in ALIGN phase, if ball becomes missing again go back to SEARCH
        else if (phase == Phase::ALIGN) {''',
'''        else if (phase == Phase::SIDESTEP) {
            target_vx = 0.0f;
            target_vy = STRAFE_SIGN * SIDESTEP_SPEED; // CONFIRM sign on hardware
            target_wz = 0.0f;
            sidestep_counter++;
            std::cout << "[fetch] SIDESTEP step " << sidestep_counter << "/" << SIDESTEP_STEPS << "\\n";
            if (sidestep_counter >= SIDESTEP_STEPS) {
                phase = Phase::WALK_PAST;
                walk_past_counter = 0;
                std::cout << "[fetch] sidestep complete -> WALK_PAST\\n";
            }
        }
        else if (phase == Phase::WALK_PAST) {
            target_vx = WALK_PAST_SPEED;
            target_vy = 0.0f;
            target_wz = 0.0f;
            walk_past_counter++;
            std::cout << "[fetch] WALK_PAST step " << walk_past_counter << "/" << WALK_PAST_STEPS << "\\n";
            if (walk_past_counter >= WALK_PAST_STEPS) {
                phase = Phase::TURN_AROUND;
                turn_around_counter = 0;
                std::cout << "[fetch] walk past complete -> TURN_AROUND\\n";
            }
        }
        else if (phase == Phase::TURN_AROUND) {
            target_vx = 0.0f;
            target_vy = 0.0f;
            target_wz = TURN_AROUND_RATE; // CONFIRM sign/direction on hardware
            turn_around_counter++;
            std::cout << "[fetch] TURN_AROUND step " << turn_around_counter << "/" << TURN_AROUND_STEPS << "\\n";
            if (turn_around_counter >= TURN_AROUND_STEPS) {
                target_wz = 0.0f;
                phase = Phase::SIDESTEP_RECENTER;
                recenter_counter = 0;
                std::cout << "[fetch] turn around complete -> SIDESTEP_RECENTER\\n";
            }
        }
        else if (phase == Phase::SIDESTEP_RECENTER) {
            target_vx = 0.0f;
            // Same body-frame sign as the initial SIDESTEP. After a real ~180 turn this should
            // cancel the step-3 world-frame offset -- CONFIRM this actually holds on hardware.
            target_vy = STRAFE_SIGN * RECENTER_SIDESTEP_SPEED;
            target_wz = 0.0f;
            recenter_counter++;
            std::cout << "[fetch] SIDESTEP_RECENTER step " << recenter_counter << "/" << RECENTER_SIDESTEP_STEPS << "\\n";
            if (recenter_counter >= RECENTER_SIDESTEP_STEPS) {
                phase = Phase::ALIGN;
                std::cout << "[fetch] recenter complete -> ALIGN\\n";
            }
        }
        // Only run if currently in ALIGN phase, if ball becomes missing again go back to SEARCH
        else if (phase == Phase::ALIGN) {'''
))

edits.append((
'''                if (!person.found) {
                    // strafe sideways at constantly speed to search for marker while circling the ball
                    target_vy = STRAFE_SIGN * CIRCLE_STRAFE_SPEED;
                    // ARE we aligned with ball and marker set to NO
                    align_coarse_confirm.confirm(false, ALIGN_COARSE_CONFIRM_FRAMES);

                // if the marker IS visible in frame
                } else {''',
'''                if (!person.found) {
                    // no-op: previously strafed at full CIRCLE_STRAFE_SPEED to search while orbiting;
                    // not needed now that SIDESTEP_RECENTER lands the robot roughly facing the marker.
                    target_vy = 0.0f;
                    align_coarse_confirm.confirm(false, ALIGN_COARSE_CONFIRM_FRAMES);

                // if the marker IS visible in frame
                } else {'''
))

for i, (old, new) in enumerate(edits, 1):
    count = src.count(old)
    if count != 1:
        print(f"EDIT {i}: expected 1 match, found {count}. Aborting.", file=sys.stderr)
        sys.exit(1)
    src = src.replace(old, new)

open(path, "w").write(src)
print("All 3 edits applied.")
