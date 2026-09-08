// Real-hardware Go2 fetch controller.
// Reuses the existing, already-validated FSM/State_RLBase/autopilot machinery
// from main.cpp unchanged. The only new part is the vision loop below.
//
// NOTE: HSV threshold values below are PLACEHOLDERS. They WILL need live
// recalibration against real camera footage under real lighting.
//
// SESSION CHANGES (this pass):
//   1. New Phase::INITIAL_SETTLE -- after first ball detection, hold position
//      for INITIAL_SETTLE_STEPS iterations (~6s at VISION_HZ) before APPROACH_BALL
//      starts moving, so a rolling/bouncing ball has time to stop.
//   2. line_heading: snapshotted (from live IMU yaw) at the moment the robot
//      commits to SIDESTEP. SIDESTEP / WALK_PAST / SIDESTEP_RECENTER now
//      closed-loop correct target_wz to hold that heading instead of assuming
//      wz=0 holds straight. TURN_AROUND now targets (line_heading + pi) directly
//      via measured yaw instead of a fixed step count.
//   3. TURN_DIRECTION_SIGN is a single flip point if the turn comes out backward
//      on hardware -- see note at its declaration below. UNCONFIRMED, first
//      hardware test of this logic should watch the printed diff shrink, not grow.
//   4. ALIGN_HEADING (a pure-rotation align step before each translation phase) was
//      tried, then REMOVED per explicit design decision -- it added a confusing
//      multi-second stall on hardware. TURN_AROUND remains the only realignment step.
//      SIDESTEP/WALK_PAST/SIDESTEP_RECENTER/FINAL_PUSH are pure open-loop translation,
//      wz flat at 0 throughout, no heading correction at all.

#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "isaaclab/utils/autopilot.h"
#include <unitree/robot/go2/video/video_client.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <cmath>

// BLACK & WHITE MARKER
const int ARUCO_DICT_TYPE = cv::aruco::DICT_4X4_50;
const int VEST_MARKER_ID = 0;

// CONFIRMATION CAMERA FRAME PARAMETERS                    * POSSIBLE ERRORS * - some too strict?
const int LOST_BALL_CONFIRM_FRAMES = 5;
const int APPROACH_CLOSE_CONFIRM_FRAMES = 1;
const int ALIGN_COARSE_CONFIRM_FRAMES = 5;      // 5?
const int FINEALIGN_CONFIRM_FRAMES = 3;

// LAST STRETCH of FETCH PARAMETERS
// NEW: blind charge back along line_heading+pi after SIDESTEP_RECENTER -- no marker/ball
// tracking anymore, just heading-locked dead reckoning. 10 steps is a first guess.
const int FINAL_PUSH_STEPS = 10;
const float FINAL_PUSH_SPEED = 1.0f;

// SIDESTEP / WALK_PAST / TURN_AROUND / SIDESTEP_RECENTER PARAMETERS (untested guesses, calibrate on hardware)
const float SIDESTEP_SPEED = 0.35f;             // Was 0.2 (before that 0.4). 0.4 was borderline unsafe (near
                                                  // fall), 0.2 felt too slow/cautious -- 0.35 as a middle ground.
                                                  // NEEDS CALIBRATION.
const int SIDESTEP_STEPS = 8;                    // Was 4 (before that 8, 5). Doubled again -- 4 steps at 0.2
                                                  // m/s was confirmed too short on hardware (basically one step).
                                                  // Speed unchanged at 0.2 for the safety margin established
                                                  // earlier. NEEDS CALIBRATION.
const float WALK_PAST_SPEED = 1.0f;              // forward speed while blindly passing the ball
const int WALK_PAST_STEPS = 7;                   // Was 5 (before that 16, 8, 5, 3). Slight increase per
                                                  // direct hardware feedback -- not doubled like SIDESTEP,
                                                  // just a modest bump. NEEDS CALIBRATION.
const float TURN_AROUND_RATE = 0.8f;             // max turn speed magnitude for the 180 (now yaw-target-driven, see Phase::TURN_AROUND)
const float RECENTER_SIDESTEP_SPEED = 0.35f;     // Was 0.2 (before that 0.15, 0.4). Matches SIDESTEP_SPEED's
                                                  // new value directly.
const int RECENTER_SIDESTEP_STEPS = 8;           // Was 4 (before that 4, 13, 21, 8). Doubled to match
                                                  // SIDESTEP_STEPS (8) again -- still covers the same distance
                                                  // as the initial sidestep (0.2*8=1.6 both ways). NEEDS CALIBRATION.

// NEW: pause after first detecting the ball, before APPROACH_BALL starts moving,
// so a rolling/bouncing ball has time to settle. VISION_HZ is 2.0 (defined below),
// so 12 iterations = ~6s. If VISION_HZ changes, this iteration count must be re-derived.
const int INITIAL_SETTLE_STEPS = 12;

// NEW: heading-lock parameters. During SIDESTEP/WALK_PAST/SIDESTEP_RECENTER, this
// is a proportional correction toward line_heading instead of assuming wz=0 holds
// straight (it doesn't -- slew-limiter carryover from the previous phase bleeds in).
// SUPERSEDED (kept declared, unused): these drove a continuous heading correction
// DURING translation (SIDESTEP/WALK_PAST/SIDESTEP_RECENTER/FINAL_PUSH all commanding
// vy or vx and wz at once). Replaced by Phase::ALIGN_HEADING, which corrects heading
// as its own separate pure-rotation step before each translation begins -- per
// explicit design decision after a hardware fall suspected to be caused by a large,
// sudden wz correction landing mid-SIDESTEP.
const float HEADING_HOLD_KP = 1.2f;         // rad/s per rad of heading error
const float HEADING_HOLD_MAX_WZ = 0.5f;     // clamp on the correction itself

// NEW: TURN_AROUND is now yaw-target-driven (target = line_heading + pi) rather
// than a fixed step count, replacing the old "6 steps ~150deg, needs 7" guessing.
const float TURN_KP = 1.5f;                 // rad/s per rad of yaw error
const float TURN_MIN_RATE = 0.15f;          // floor so it doesn't crawl to a stop near the target
const float TURN_DONE_TOLERANCE = 0.06f;    // ~3.4 deg, counts as "arrived"
const int TURN_AROUND_MAX_STEPS = 25;       // safety cap: if diff is growing instead of
                                             // shrinking, TURN_DIRECTION_SIGN is backward --
                                             // this stops it from spinning indefinitely.
// SUPERSEDED (kept declared, unused): was the safety cap for Phase::ALIGN_HEADING,
// which was removed -- see note at the Phase enum declaration.
const int ALIGN_HEADING_MAX_STEPS = 10;
// UNCONFIRMED ON HARDWARE: if the printed [turn-debug] diff grows instead of shrinks
// during the first live test, flip this to -1.0f and rebuild.
const float TURN_DIRECTION_SIGN = 1.0f;

// LIBRARIES
#include <chrono>
#include <thread>

// CONNECTS FETCH CTR TO ROBOT CONTROL SYSTEM (DON'T TOUCH, NO ERRORS)
std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

// NEW: slew-limiter memory, now file-scope so settle_pause() can reset it. Previously
// this was a `static` local inside the main loop's slew block, invisible to
// settle_pause() -- so settle_pause()'s hard-zero command wasn't reflected here, and
// the next phase's slew ramp incorrectly assumed continuity from the PRE-pause
// velocity, reintroducing old momentum right when a phase transition wanted a clean
// stop. This is what carried the robot through the ball during SIDESTEP tonight.
float g_prev_vx = 0.0f, g_prev_vy = 0.0f, g_prev_wz = 0.0f;

const float MAX_FORWARD_SPEED = 1.0f;       // First Approach to Ball Speed
const float TURN_GAIN = 1.0f;               // TURN LEFT & RIGHT (-1 -> 1) while first approaching ball
const float TURN_SIGN = -1.0f;
const float BALL_TURN_GAIN = 1.0f;          // Used in ALIGN / FINE ALIGN: centering ball while close to the ball
                                             // NEW: lowered from 3.0 -- tonight's first-ever ALIGN run showed the ball
                                             // offset oscillating hard between opposite frame edges rather than
                                             // converging (classic high-gain overshoot). This is a conservative first
                                             // guess, not a confirmed fix -- expect to retune further next run.
const float SEARCH_SPIN_RATE = 0.0f;        // If ball is not in frame, do not spin searching for it (ethernet cord safety)

// Reverted to 180,000: the YOLO-confidence-crash issue only applied to the old continuous ALIGN tracking. SIDESTEP/WALK_PAST/TURN_AROUND are blind now, so real closeness to the ball matters more than YOLO confidence at the trigger moment. NEEDS CALIBRATION.
const double BALL_CLOSE_AREA_PX = 100000.0; // Was 180000. Tonight's actual sequence right before contact was
                                             // 42460 -> 72779 -> 147768 -> 275377(contact) -- area roughly doubles
                                             // per 0.5s tick near the end, so 180000 wasn't crossed until the ball
                                             // was already essentially touching the robot. 100000 sits just above
                                             // the 72779 frame and below 147768, so it should trigger one full tick
                                             // earlier than tonight -- more standoff, no change to approach speed.
                                             // NEEDS CALIBRATION -- watch the printed ball_area at the actual
                                             // trigger frame next run; if it's still touching, this needs to drop
                                             // further, but not blindly toward 65000 (a much earlier session found
                                             // area plateaued ~55-64k and never crossed that threshold at all --
                                             // that was a different setup, but worth keeping in mind before going
                                             // that low again).
const double DECEL_START_AREA_PX = BALL_CLOSE_AREA_PX * 0.4; // start slowing down well before reaching the ball
const float MIN_APPROACH_SPEED = 1.0f;                      // flat approach speed (no deceleration) -- equal to MAX_FORWARD_SPEED; NEEDS RECHECK if limping persists under autopilot

const double COARSE_TOLERANCE_PX = 120.0;   // Used in ALIGN: 120 pixel pillow room for if ball and marker are centered
const double FINE_TOLERANCE_PX = 30.0;      // Used in FINE ALIGN: 30 pixels of tolerance, 15 on either side of halved camera line
const float FINE_ALIGN_CREEP_SPEED = 0.08f; // Used in FINE ALIGN: inch forward to make alignment easier

// STRAFING AROUND BALL
const float CIRCLE_STRAFE_SPEED = 0.32f;    // Circle around the ball speed
const float STRAFE_SIGN = 1.0f;             // USED in ALIGN: speed moving left to right without turning
const float STRAFE_GAIN = 0.008f;           // Once person + marker is visible, scalar for how strong robot corrects its strafe

// CAMERA DIMENSIONS + CUT IN HALF LINE
const int CAM_WIDTH = 1920;
const int CAM_HEIGHT = 1080;
const int CENTER_X = CAM_WIDTH / 2;

// CHECKING CAMERA FRAMES
const double VISION_HZ = 2.0;       // Used in SEARCH, APPROACH_BALL, WALK_THROUGH, FINAL PUSH: Checks camera 2 times /sec
const double ALIGN_VISION_HZ = 6.0; // Used in ALIGN / FINE ALIGN:                              Checks camera 6 times /sec

// SHARED RESULT STRUCT: used by both find_ball and find_vest to report
// whether something was detected, its center position, and its area
struct BlobResult {
    bool found = false;
    double cx = 0, cy = 0, area = 0;
};

// TRACKS N-CONSECUTIVE-FRAMES-TRUE FOR DEBOUNCING PHASE TRANSITIONS
struct FrameConfirm {
    int count = 0;
    bool confirm(bool condition, int required_frames) {
        count = condition ? count + 1 : 0;
        return count >= required_frames;
    }
};

// NEW: reads live yaw (heading) from the IMU quaternion on FSMState::lowstate.
// Confirmed quaternion index order from unitree_articulation.h: it's passed
// positionally into Eigen::Quaternionf(w, x, y, z), so index 0 is w.
float get_current_yaw()
{
    const auto& q = FSMState::lowstate->msg_.imu_state().quaternion(); // [w, x, y, z]
    float w = q[0], x = q[1], y = q[2], z = q[3];
    return std::atan2(2.0f * (w * z + x * y), 1.0f - 2.0f * (y * y + z * z));
}

// NEW: signed shortest-path angle difference (target - current), wrapped to [-pi, pi]
float angle_diff(float target, float current)
{
    float d = target - current;
    while (d > (float)M_PI) d -= 2.0f * (float)M_PI;
    while (d < -(float)M_PI) d += 2.0f * (float)M_PI;
    return d;
}

// FINDS MARKER IN CAMERA FRAME, OUTPUTS POSITION / SIZE
// USED in ALIGN, FINE_ALIGN, WALK_THROUGH
// Also called in SEARCH, APPROACH_BALL as well (for the shortcut specifically, wasted work during final deployment)
// SETTLE / STOP-BETWEEN-PHASES: brief full stop between blind sub-phases so each maneuver has a
// clean, distinct start instead of one command blending directly into the next.
const float SETTLE_PAUSE_SEC = 0.5f; // NEEDS CALIBRATION
void settle_pause()
{
    isaaclab::autopilot::set(0.0f, 0.0f, 0.0f);
    // NEW: the robot really is at zero now -- make the slew limiter's memory agree,
    // so the next phase ramps from true zero instead of resuming the pre-pause velocity.
    g_prev_vx = 0.0f; g_prev_vy = 0.0f; g_prev_wz = 0.0f;
    std::cout << "[fetch] settling...\n";
    std::this_thread::sleep_for(std::chrono::milliseconds((int)(SETTLE_PAUSE_SEC * 1000)));
}
BlobResult find_vest(const cv::Mat& frame, cv::aruco::ArucoDetector& detector)
{                                                                                       // OUTPUTS POSITION / SIZE
    BlobResult result;                                  // Emtpy container for "result", (found, cx, cy, area)
    std::vector<int> ids;                               // Contrainer holding all ids in frame
    std::vector<std::vector<cv::Point2f>> corners;      // Container holding dim of all markers

    detector.detectMarkers(frame, corners, ids);        // OpenCV ArUco detection call, scans frame, finds markers

    for (size_t i = 0; i < ids.size(); ++i) {           // loops through every marker found, should just find VEST_MARKER_ID = 0
        if (ids[i] != VEST_MARKER_ID) continue;

        cv::Point2f center(0, 0);                       // find center point of marker
        for (const auto& pt : corners[i]) center += pt;
        center *= 0.25f;

        cv::Rect box = cv::boundingRect(corners[i]);    // dim. of rectangle that contains all 4 corners of marker
        result.found = true;                            // whether the marker is found
        result.cx = center.x;                           // center of marker coordinates
        result.cy = center.y;
        result.area = (double)box.width * box.height;   // area (width x height) of marker, larger area = marker is closer, etc.
        break;
    }

    return result;                                      // OUTPUTS POSITION / SIZE of MARKER
}

// NEURAL NETWORK FOR BALL DETECTION                    * POSSIBLE ERRORS * - Make custom model trained on specific BALL
const int YOLO_INPUT_SIZE = 640;        // 640 x 640 pixels, resized later to actual camera frame: 1920 x 1080
const float BALL_CONF_THRESHOLD = 0.10f; // The model must be 20% sure or more whether the ball is in frame
const int SPORTS_BALL_CLASS_ID = 0;    // #32 = Sports Ball
const std::string YOLO_MODEL_PATH = "/home/miro/unitree_rl_lab/deploy/robots/go2/ball_yolov8n.onnx"; // path to model weights file


// YOLO CAMERA FRAMES DETECTION                         * POSSIBLE ERRORS * - Make custom model trained on specific BALL
BlobResult find_ball(cv::dnn::Net& net, const cv::Mat& frame)
{
    BlobResult result;      // empty container for output

    // RESIZING FRAME FOR YOLO BALL DETECTION
    float r = std::min((float)YOLO_INPUT_SIZE / frame.cols, (float)YOLO_INPUT_SIZE / frame.rows);
    int new_w = (int)std::round(frame.cols * r);
    int new_h = (int)std::round(frame.rows * r);
    int pad_x = (YOLO_INPUT_SIZE - new_w) / 2;
    int pad_y = (YOLO_INPUT_SIZE - new_h) / 2;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_w, new_h));
    cv::Mat padded(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, frame.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(pad_x, pad_y, new_w, new_h)));

    cv::Mat blob;
    cv::dnn::blobFromImage(padded, blob, 1.0/255.0, cv::Size(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE),
                            cv::Scalar(), true, false);
    net.setInput(blob);
    cv::Mat output = net.forward();

    cv::Mat output_reshaped = output.reshape(1, 5);

    // Undo letterbox: map box coords from padded 640x640 space back to original frame
    float inv_scale = 1.0f / r;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    float max_conf_seen = 0.0f;

    // for YOLO, loops through all detections and keeps only ones that look like SPORTS BALL
    for (int i = 0; i < output_reshaped.cols; ++i) {
        float conf = output_reshaped.at<float>(4 + SPORTS_BALL_CLASS_ID, i);
        if (conf > max_conf_seen) max_conf_seen = conf;
        if (conf < BALL_CONF_THRESHOLD) continue;

        float cx = output_reshaped.at<float>(0, i);
        float cy = output_reshaped.at<float>(1, i);
        float w  = output_reshaped.at<float>(2, i);
        float h  = output_reshaped.at<float>(3, i);

        int left = (int)(((cx - w / 2.0f) - pad_x) * inv_scale);
        int top  = (int)(((cy - h / 2.0f) - pad_y) * inv_scale);
        int width  = (int)(w * inv_scale);
        int height = (int)(h * inv_scale);

        boxes.emplace_back(left, top, width, height);
        confidences.push_back(conf);
    }

    // if YOLO can't detect any SPORT BALLS in frame
    if (boxes.empty()) {
        std::cout << "[ball-debug] no candidate boxes above conf=" << BALL_CONF_THRESHOLD
                   << " (max confidence seen this frame: " << max_conf_seen << ")\n";
        return result;
    }

    // Handle overlapping boxes around the same ball
    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, confidences, BALL_CONF_THRESHOLD, 0.45f, nms_indices); // if two detections for same ball overlap, combine
    if (nms_indices.empty()) {
        std::cout << "[ball-debug] " << boxes.size() << " candidate(s) found but NMS eliminated all\n";
        return result;
    }

    // DEBUG PRINT STATEMENT: confirming how many SPORTS BALL CANDIDATES survived
    std::cout << "[ball-debug] " << boxes.size() << " candidate(s), " << nms_indices.size() << " survived NMS\n";

    // Picks highest conf. box that was capturing the ball to stick with
    int best_idx = nms_indices[0];
    float best_conf = confidences[best_idx];
    for (int idx : nms_indices) {
        if (confidences[idx] > best_conf) { best_conf = confidences[idx]; best_idx = idx; }
    }

    // Takes the winning ball detection frame and converts to format the rest of the code can work with
    cv::Rect best_box = boxes[best_idx];
    result.found = true;
    result.cx = best_box.x + best_box.width / 2.0;
    result.cy = best_box.y + best_box.height / 2.0;
    result.area = (double)best_box.width * best_box.height;
    return result;
}

// SAFETY CHECK TO MAKE SURE NOTHING ELSE IS TRYING TO CONTROL THE ROBOT
void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}



// MAIN______________________________________________________________________________________
int main(int argc, char** argv)
{
    // Creates variable map, parses command like --network eth0
    auto vm = param::helper(argc, argv);

    // Prints startup
    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     Go2 Fetch Controller \n";

    // Connects robot to computer via ethernet connection
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    // Sets up the different modes the robot can be in, walking, standing
    init_fsm_state();

    // Create the brain that controls these modes/states & turn brain on
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    // limp legs -> standup
    std::cout << "[fetch] Waiting 2s in Passive before standing up...\n";
    sleep(2);
    std::cout << "[fetch] Forcing FixStand...\n";
    fsm->forceState(2);
    sleep(4);

    // Switch to walking mode, controlled by policy
    std::cout << "[fetch] Forcing Velocity (policy control)...\n";
    fsm->forceState(3);
    sleep(1);

    // Set up video client to pull frames from
    unitree::robot::go2::VideoClient video_client;
    video_client.Init();

    // Loads YOLO ball detection + run the model on GPU so its fast enough to make video
    cv::dnn::Net ball_net = cv::dnn::readNetFromONNX(YOLO_MODEL_PATH);
    ball_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    ball_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    // Loads set of known markers + default settings + creates marker detection
    // and now ready to see the marker in each frame
    cv::aruco::Dictionary aruco_dict = cv::aruco::getPredefinedDictionary(ARUCO_DICT_TYPE);
    cv::aruco::DetectorParameters aruco_params = cv::aruco::DetectorParameters();
    cv::aruco::ArucoDetector aruco_detector(aruco_dict, aruco_params);

    // POSSIBLE STAGES
    // NEW: INITIAL_SETTLE inserted between SEARCH (first detection) and APPROACH_BALL
    // (moving toward it), so a rolling/bouncing ball has time to stop first.
    // NEW: ALIGN_HEADING (a pure-rotation correction step before each translation
    // phase) was tried and then REMOVED per explicit design decision -- it added a
    // confusing multi-second stall (one run hit the max-step cap without closing the
    // error) right between SIDESTEP and WALK_PAST, making the whole sequence look like
    // nothing was moving. TURN_AROUND remains the only realignment step, since turning
    // around is the actual task there, not a correction to something else. SIDESTEP,
    // WALK_PAST, SIDESTEP_RECENTER, and FINAL_PUSH are all pure open-loop translation
    // now -- wz flat at 0, no heading correction at all, trusting step-count/speed
    // calibration plus the settle_pause/slew fix to stay reasonably straight.
    enum class Phase { SEARCH, INITIAL_SETTLE, APPROACH_BALL, SIDESTEP, WALK_PAST, TURN_AROUND, SIDESTEP_RECENTER, FINAL_PUSH, DONE };
    // START: Search for ball in frame
    Phase phase = Phase::SEARCH;

    // ALL COUNTERS for counting successful frames in a row
    FrameConfirm lost_ball_search_confirm;
    FrameConfirm approach_close_confirm;

    // Steps taken during final push of ball
    int final_push_counter = 0;
    int sidestep_counter = 0;
    int walk_past_counter = 0;
    int turn_around_counter = 0;
    int recenter_counter = 0;
    int initial_settle_counter = 0; // NEW

    // NEW: heading captured the moment the robot commits to SIDESTEP -- used only by
    // TURN_AROUND now (targets line_heading + pi) since ALIGN_HEADING was removed.
    float line_heading = 0.0f;

    // 3 MOVEMENT COMMANDS: forward, sideways, & turn speed
    float target_vx = 0.0f, target_vy = 0.0f, target_wz = SEARCH_SPIN_RATE;

    // 1st COMMANDS to ROBOT: Don't move yet
    isaaclab::autopilot::set(target_vx, target_vy, target_wz);
    // How often to check camera
    auto vision_period = std::chrono::milliseconds((int)(1000.0 / VISION_HZ));
    // NOTE: ALIGN_VISION_HZ / align_period removed -- was only used by ALIGN/FINE_ALIGN,
    // which no longer exist in the active flow. ALIGN_VISION_HZ constant left declared
    // above (harmless, unused) in case this gets re-enabled later.

    //_______________________________________________________________________________________________
    // STARTS FETCH CODE
    std::cout << "[fetch] Starting fetch loop. Ctrl+C to stop.\n";
    // MAIN LOOP
    while (phase != Phase::DONE)
    {
        // NEW: ALIGN/FINE_ALIGN removed, so the faster align_period polling rate is no
        // longer used anywhere -- every phase now runs at the standard vision_period.
        auto period = vision_period;
        std::this_thread::sleep_for(period);

        // Grabs jpeg photo from camera
        std::vector<uint8_t> jpeg_bytes;
        int32_t ret = video_client.GetImageSample(jpeg_bytes);

        // If grabbing photo failed, print error
        if (ret != 0 || jpeg_bytes.empty()) {
            std::cout << "[warning] camera frame grab failed, ret=" << ret << "\n";
            continue;
        }
        // Turn jpeg into image software can work with
        cv::Mat frame = cv::imdecode(jpeg_bytes, cv::IMREAD_COLOR);
        if (frame.empty()) continue;
        // Print camera width and height to confirm camera is working
        std::cout << "[debug] frame size: " << frame.cols << "x" << frame.rows << "\n";

        // run YOLO model on usable frame to find the ball, outputs yes/no found, and where it is
        BlobResult ball = find_ball(ball_net, frame);
        // run YOLO model on usable frame to find the marker, outputs yes/no found, and where it is
        BlobResult person = find_vest(frame, aruco_detector);
        // NEW: current measured yaw, used by heading-lock phases below
        float current_yaw = get_current_yaw();
        // Clone image for drawing on
        cv::Mat display = frame.clone();
        // Draw cut in half camera line down the middle
        cv::line(display, cv::Point(CENTER_X, 0), cv::Point(CENTER_X, CAM_HEIGHT), cv::Scalar(0,0,0), 1);
        // if they are found, draw "GREEN +" on ball and "Orange +" on marker
        if (ball.found) cv::drawMarker(display, cv::Point((int)ball.cx, (int)ball.cy), cv::Scalar(0,255,0), cv::MARKER_CROSS, 20, 2);
        if (person.found) cv::drawMarker(display, cv::Point((int)person.cx, (int)person.cy), cv::Scalar(0,140,255), cv::MARKER_CROSS, 20, 2);

        // Show image, with the "+" on it & cut in half screen
        cv::imshow("Go2 Fetch - live view", display);
        cv::waitKey(1);

        // MAIN LOGIC OF FETCH_______________________________________________________________________________

        if (phase == Phase::SEARCH) {
            if (!ball.found) {
                // Don't move
                target_vx = 0.0f; target_vy = 0.0f; target_wz = 0.0f;
                approach_close_confirm.confirm(false, APPROACH_CLOSE_CONFIRM_FRAMES);
                if (lost_ball_search_confirm.confirm(true, LOST_BALL_CONFIRM_FRAMES)) {
                    phase = Phase::SEARCH;
                    target_wz = SEARCH_SPIN_RATE;
                }
            } else {
                // NEW: ball just appeared -- don't charge yet, go settle first.
                lost_ball_search_confirm.confirm(false, LOST_BALL_CONFIRM_FRAMES);
                target_vx = 0.0f; target_vy = 0.0f; target_wz = 0.0f;
                phase = Phase::INITIAL_SETTLE;
                initial_settle_counter = 0;
                std::cout << "[fetch] ball detected -> INITIAL_SETTLE\n";
            }
            std::cout << "[fetch] phase=" << (int)phase << " target=(" << target_vx << "," << target_vy << "," << target_wz << ") ball_area=" << ball.area << "\n";
        }
        // NEW PHASE: hold still for INITIAL_SETTLE_STEPS iterations so a
        // rolling/bouncing ball has time to stop before the robot commits to it.
        else if (phase == Phase::INITIAL_SETTLE) {
            target_vx = 0.0f; target_vy = 0.0f; target_wz = 0.0f;
            if (!ball.found) {
                // Ball rolled out of frame during settle -- don't guess, go back to searching.
                std::cout << "[fetch] ball lost during INITIAL_SETTLE -> SEARCH\n";
                phase = Phase::SEARCH;
                initial_settle_counter = 0;
            } else {
                initial_settle_counter++;
                std::cout << "[fetch] INITIAL_SETTLE step " << initial_settle_counter << "/" << INITIAL_SETTLE_STEPS << "\n";
                if (initial_settle_counter >= INITIAL_SETTLE_STEPS) {
                    phase = Phase::APPROACH_BALL;
                    std::cout << "[fetch] settle complete -> APPROACH_BALL\n";
                }
            }
        }
        else if (phase == Phase::APPROACH_BALL) {
            if (!ball.found) {
                target_vx = 0.0f; target_vy = 0.0f; target_wz = 0.0f;
                approach_close_confirm.confirm(false, APPROACH_CLOSE_CONFIRM_FRAMES);
                if (lost_ball_search_confirm.confirm(true, LOST_BALL_CONFIRM_FRAMES)) {
                    phase = Phase::SEARCH;
                    target_wz = SEARCH_SPIN_RATE;
                }
            } else {
                lost_ball_search_confirm.confirm(false, LOST_BALL_CONFIRM_FRAMES);
                double offset = (ball.cx - CENTER_X) / (CAM_WIDTH / 2.0);
                target_wz = std::clamp((float)(TURN_SIGN * TURN_GAIN * offset), -1.0f, 1.0f);
                if (ball.area < DECEL_START_AREA_PX) {
                    target_vx = MAX_FORWARD_SPEED;
                } else {
                    double t = (ball.area - DECEL_START_AREA_PX) / (BALL_CLOSE_AREA_PX - DECEL_START_AREA_PX);
                    t = std::clamp(t, 0.0, 1.0);
                    target_vx = MAX_FORWARD_SPEED - (float)t * (MAX_FORWARD_SPEED - MIN_APPROACH_SPEED);
                }
                target_vy = 0.0f;
                if (approach_close_confirm.confirm(ball.area >= BALL_CLOSE_AREA_PX, APPROACH_CLOSE_CONFIRM_FRAMES)) {
                    // NEW: snapshot the line here -- this is the last moment heading
                    // is deliberately chosen before the blind maneuver sequence.
                    line_heading = current_yaw;
                    phase = Phase::SIDESTEP;
                    sidestep_counter = 0;
                    std::cout << "[fetch] close to ball -> SIDESTEP (line_heading=" << line_heading << ")\n";
                    settle_pause();
                }
            }
            std::cout << "[fetch] phase=" << (int)phase << " target=(" << target_vx << "," << target_vy << "," << target_wz << ") ball_area=" << ball.area << "\n";
        }
        // NEW: generic pure-rotation align step, same yaw-servo pattern as TURN_AROUND
        // (zero vx, zero vy, only wz), reused before every translation phase so the
        // robot aligns first, then moves -- never both at once. align_target_yaw and
        // align_next_phase are set by whichever phase transitions into this one.
        // REMOVED per explicit design decision: this added a confusing multi-second
        // stall (one run hit ALIGN_HEADING_MAX_STEPS without closing the error) sitting
        // right between SIDESTEP and WALK_PAST, making the sequence look like nothing
        // was moving. TURN_AROUND remains the only realignment step now.
        // REWRITTEN: pure lateral motion, wz flat at 0, no align step before or after --
        // straight from APPROACH_BALL's line_heading capture into this.
        else if (phase == Phase::SIDESTEP) {
            target_vx = 0.0f;
            target_vy = -STRAFE_SIGN * SIDESTEP_SPEED; // CONFIRMED on hardware: +STRAFE_SIGN went left, negated to get right
            target_wz = 0.0f;
            sidestep_counter++;
            std::cout << "[fetch] SIDESTEP step " << sidestep_counter << "/" << SIDESTEP_STEPS << "\n";
            if (sidestep_counter >= SIDESTEP_STEPS) {
                phase = Phase::WALK_PAST;
                walk_past_counter = 0;
                std::cout << "[fetch] sidestep complete -> WALK_PAST\n";
                settle_pause();
            }
        }
        // REWRITTEN: pure forward motion, wz flat at 0 (see SIDESTEP note above).
        else if (phase == Phase::WALK_PAST) {
            target_vx = WALK_PAST_SPEED;
            target_vy = 0.0f;
            target_wz = 0.0f;
            walk_past_counter++;
            std::cout << "[fetch] WALK_PAST step " << walk_past_counter << "/" << WALK_PAST_STEPS << "\n";
            if (walk_past_counter >= WALK_PAST_STEPS) {
                phase = Phase::TURN_AROUND;
                turn_around_counter = 0;
                std::cout << "[fetch] walk past complete -> TURN_AROUND\n";
                settle_pause();
            }
        }
        // Was a fixed step count (6, known to undershoot to ~150deg). Now targets
        // (line_heading + pi) directly via measured yaw and stops on arrival, with a
        // max-iteration safety cap in case TURN_DIRECTION_SIGN is backward on hardware
        // (diff would grow instead of shrink). Already a pure-rotation phase -- this is
        // the pattern ALIGN_HEADING above reuses.
        else if (phase == Phase::TURN_AROUND) {
            float target_yaw = line_heading + (float)M_PI;
            while (target_yaw > (float)M_PI) target_yaw -= 2.0f * (float)M_PI;
            while (target_yaw < -(float)M_PI) target_yaw += 2.0f * (float)M_PI;
            float diff = angle_diff(target_yaw, current_yaw);
            turn_around_counter++;

            target_vx = 0.0f;
            target_vy = 0.0f;

            if (std::abs(diff) < TURN_DONE_TOLERANCE || turn_around_counter >= TURN_AROUND_MAX_STEPS) {
                target_wz = 0.0f;
                phase = Phase::SIDESTEP_RECENTER;
                recenter_counter = 0;
                std::cout << "[fetch] turn around complete (diff=" << diff
                          << ", steps=" << turn_around_counter << ") -> SIDESTEP_RECENTER\n";
                if (turn_around_counter >= TURN_AROUND_MAX_STEPS && std::abs(diff) >= TURN_DONE_TOLERANCE) {
                    std::cout << "[fetch] WARNING: TURN_AROUND hit max step cap without closing diff -- "
                                 "check TURN_DIRECTION_SIGN, it may be backward\n";
                }
                settle_pause();
            } else {
                float mag = std::clamp(std::abs(diff) * TURN_KP, TURN_MIN_RATE, TURN_AROUND_RATE);
                target_wz = TURN_DIRECTION_SIGN * (diff > 0 ? 1.0f : -1.0f) * mag;
            }
            std::cout << "[fetch] TURN_AROUND step " << turn_around_counter << " target_yaw=" << target_yaw
                      << " current_yaw=" << current_yaw << " diff=" << diff << " wz=" << target_wz << "\n";
        }
        // REWRITTEN: pure lateral motion, wz flat at 0 (see SIDESTEP note above). No
        // longer needs its own heading-hold block -- TURN_AROUND already leaves the
        // robot aligned to line_heading+pi, and this phase trusts that.
        else if (phase == Phase::SIDESTEP_RECENTER) {
            target_vx = 0.0f;
            // Same body-frame sign as the initial SIDESTEP. After a real ~180 turn this should
            // cancel the step-3 world-frame offset -- CONFIRM this actually holds on hardware.
            target_vy = -STRAFE_SIGN * RECENTER_SIDESTEP_SPEED; // same empirical fix as SIDESTEP -- CONFIRM on hardware
            target_wz = 0.0f;
            recenter_counter++;
            std::cout << "[fetch] SIDESTEP_RECENTER step " << recenter_counter << "/" << RECENTER_SIDESTEP_STEPS << "\n";
            if (recenter_counter >= RECENTER_SIDESTEP_STEPS) {
                phase = Phase::FINAL_PUSH;
                final_push_counter = 0;
                std::cout << "[fetch] recenter complete -> FINAL_PUSH\n";
                settle_pause();
            }
        }
        // Per explicit design decision: after SIDESTEP_RECENTER, the robot should
        // already be back on line_heading, facing line_heading+pi. This phase charges
        // straight back along that heading, blind, for FINAL_PUSH_STEPS iterations.
        // Pure forward motion, wz flat at 0, no align step before it -- no ball or
        // marker detection involved.
        else if (phase == Phase::FINAL_PUSH) {
            target_vx = FINAL_PUSH_SPEED;
            target_vy = 0.0f;
            target_wz = 0.0f;
            final_push_counter++;
            std::cout << "[fetch] FINAL_PUSH step " << final_push_counter << "/" << FINAL_PUSH_STEPS << " vx=" << target_vx << "\n";
            if (final_push_counter >= FINAL_PUSH_STEPS) {
                target_vx = 0.0f; target_vy = 0.0f; target_wz = 0.0f;
                phase = Phase::DONE;
                std::cout << "[fetch] final push complete -> DONE\n";
            }
        }
        // Debug, print current phase and all movement commands, runs every single loop iteration, regardless of phase
        std::cout << "[align-debug] phase=" << (int)phase << " vx=" << target_vx << " vy=" << target_vy << " wz=" << target_wz << "\n";
        // Send these movement commands to go2
        // slew-rate limit: never jump commanded velocity by more than MAX_VEL_STEP
        // per loop iteration -- an abrupt vx 1.0 -> 0 hands the policy a step
        // discontinuity it must absorb in one control cycle.
        {
            // NEW: uses g_prev_vx/vy/wz (file scope) instead of local statics, so
            // settle_pause() can reset them to a true zero baseline between phases.
            const float MAX_VEL_STEP = 0.34f;   // m/s (or rad/s) per iteration
            auto slew = [&](float target, float prev) {
                float d = target - prev;
                if (d >  MAX_VEL_STEP) d =  MAX_VEL_STEP;
                if (d < -MAX_VEL_STEP) d = -MAX_VEL_STEP;
                return prev + d;
            };
            g_prev_vx = slew(target_vx, g_prev_vx);
            g_prev_vy = slew(target_vy, g_prev_vy);
            g_prev_wz = slew(target_wz, g_prev_wz);
            std::cout << "[slew] cmd=(" << target_vx << "," << target_vy << "," << target_wz
                      << ") sent=(" << g_prev_vx << "," << g_prev_vy << "," << g_prev_wz << ")\n";
            isaaclab::autopilot::set(g_prev_vx, g_prev_vy, g_prev_wz);
        }
    }
    // End of main while loop

    // Stop moving
    isaaclab::autopilot::stop();
    cv::destroyAllWindows();
    std::cout << "[fetch] Done. Holding position.\n";
    // Keep program running without doing anything
    while (true) { sleep(1); }
    return 0;
}
