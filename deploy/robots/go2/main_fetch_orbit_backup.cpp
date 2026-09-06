// Real-hardware Go2 fetch controller.
// Reuses the existing, already-validated FSM/State_RLBase/autopilot machinery
// from main.cpp unchanged. The only new part is the vision loop below.
//
// NOTE: HSV threshold values below are PLACEHOLDERS. They WILL need live
// recalibration against real camera footage under real lighting.

#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "isaaclab/utils/autopilot.h"
#include <unitree/robot/go2/video/video_client.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

// BLACK & WHITE MARKER
const int ARUCO_DICT_TYPE = cv::aruco::DICT_4X4_50;
const int VEST_MARKER_ID = 0;

// CONFIRMATION CAMERA FRAME PARAMETERS                    * POSSIBLE ERRORS * - some too strict?
const int LOST_BALL_CONFIRM_FRAMES = 5;
const int APPROACH_CLOSE_CONFIRM_FRAMES = 3;
const int ALIGN_COARSE_CONFIRM_FRAMES = 5;      // 5?
const int FINEALIGN_CONFIRM_FRAMES = 3;
const int WALKTHROUGH_DONE_CONFIRM_FRAMES = 2;

// LAST STRETCH of FETCH PARAMETERS     
// returned to person (marker size must = 100,000 pixels^2)
const double WALKTHROUGH_ARRIVED_AREA_PX = 100000.0;    // 100,000 ? need to confirm right size
// blind walking forward after marker leaves frame from being close to the person (~3.5s at 2Hz)
const int FINAL_PUSH_STEPS = 7; 
const float FINAL_PUSH_SPEED = 0.8f;

// LIBRARIES
#include <chrono>
#include <thread>

// CONNECTS FETCH CTR TO ROBOT CONTROL SYSTEM (DON'T TOUCH, NO ERRORS)
std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

const float MAX_FORWARD_SPEED = 0.4f;       // First Approach to Ball Speed
const float TURN_GAIN = 1.0f;               // TURN LEFT & RIGHT (-1 -> 1) while first approaching ball
const float TURN_SIGN = -1.0f;
const float BALL_TURN_GAIN = 3.0f;          // Used in ALIGN / FINE ALIGN: centering ball while close to the ball
const float SEARCH_SPIN_RATE = 0.0f;        // If ball is not in frame, do not spin searching for it (ethernet cord safety)

// 180,000 may have been too close and caused YOLO to not register ball -> Trying 65,000 for further back
const double BALL_CLOSE_AREA_PX = 65000.0;                   // go2 needs to be this close before switching to align
const double DECEL_START_AREA_PX = BALL_CLOSE_AREA_PX * 0.4; // start slowing down well before reaching the ball
const float MIN_APPROACH_SPEED = 0.20f;                      // don't fully stop while still approaching, just slow down

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

// FINDS MARKER IN CAMERA FRAME, OUTPUTS POSITION / SIZE
// USED in ALIGN, FINE_ALIGN, WALK_THROUGH
// Also called in SEARCH, APPROACH_BALL as well (for the shortcut specifically, wasted work during final deployment)
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
const float BALL_CONF_THRESHOLD = 0.2f; // The model must be 20% sure or more whether the ball is in frame
const int SPORTS_BALL_CLASS_ID = 32;    // #32 = Sports Ball
const std::string YOLO_MODEL_PATH = "/home/miro/unitree_rl_lab/deploy/robots/go2/yolov8s.onnx"; // path to model weights file


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

    cv::Mat output_reshaped = output.reshape(1, 84);

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
    enum class Phase { SEARCH, APPROACH_BALL, ALIGN, FINE_ALIGN, WALK_THROUGH, FINAL_PUSH, DONE };
    // START: Search for ball in frame
    Phase phase = Phase::SEARCH;

    // ALL COUNTERS for counting successful frames in a row
    FrameConfirm lost_ball_search_confirm;
    FrameConfirm lost_ball_align_confirm;
    FrameConfirm lost_ball_finealign_confirm;
    FrameConfirm approach_close_confirm;
    FrameConfirm align_coarse_confirm;
    FrameConfirm finealign_fine_confirm;
    FrameConfirm walkthrough_done_confirm;

    // Remembers size of marker from last frame
    double last_person_area = 0.0;
    // Steps taken during final push of ball
    int final_push_counter = 0;

    // 3 MOVEMENT COMMANDS: forward, sideways, & turn speed
    float target_vx = 0.0f, target_vy = 0.0f, target_wz = SEARCH_SPIN_RATE;

    // 1st COMMANDS to ROBOT: Don't move yet
    isaaclab::autopilot::set(target_vx, target_vy, target_wz);
    // How often to check camera
    auto vision_period = std::chrono::milliseconds((int)(1000.0 / VISION_HZ));
    // USED IN ALIGN: How often to check camera (faster than regular check)
    auto align_period = std::chrono::milliseconds((int)(1000.0 / ALIGN_VISION_HZ));

    //_______________________________________________________________________________________________
    // STARTS FETCH CODE
    std::cout << "[fetch] Starting fetch loop. Ctrl+C to stop.\n";
    // MAIN LOOP
    while (phase != Phase::DONE)
    {
        // picks which camera frame capturing setting to use, faster if in ALIGN
        auto period = (phase == Phase::ALIGN || phase == Phase::FINE_ALIGN) ? align_period : vision_period;
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

        // Only run if SEARCHING for ball or APPROACHING ball
        if (phase == Phase::SEARCH || phase == Phase::APPROACH_BALL) {
            
            // IF BALL IS NOT VISIBLE IN FRAME
            if (!ball.found) {
                // Don't move
                target_vx = 0.0f; target_vy = 0.0f; target_wz = 0.0f;

                // Tell the counter that determines if the robot is close enough to the ball to switch
                // to ALIGN that we aren't close enough
                approach_close_confirm.confirm(false, APPROACH_CLOSE_CONFIRM_FRAMES);
                // Tell the counter that determines if the ball is lost that is is indeed lost
                if (lost_ball_search_confirm.confirm(true, LOST_BALL_CONFIRM_FRAMES)) {
                    // STAY IN SEARCH
                    phase = Phase::SEARCH;
                    // make sure spinrate is still 0 (don't want robot spinning around to look for ball, safety)
                    target_wz = SEARCH_SPIN_RATE;
                }
            // IF BALL IS VISIBLE IN FRAME
            } else {
                // Tell "is ball lost counter" this frame does not count as lost
                lost_ball_search_confirm.confirm(false, LOST_BALL_CONFIRM_FRAMES);
                // START Approaching ball
                phase = Phase::APPROACH_BALL;
                // Notice how much ball is off from the center, and then adjust to approach it accordingly
                double offset = (ball.cx - CENTER_X) / (CAM_WIDTH / 2.0);
                target_wz = std::clamp((float)(TURN_SIGN * TURN_GAIN * offset), -1.0f, 1.0f);
                // If ball looks small from far away
                if (ball.area < DECEL_START_AREA_PX) {
                    // Walk forward at full speed
                    target_vx = MAX_FORWARD_SPEED;
                // Else walk to the ball as if we are already close to it, AKA start walking slower
                } else {
                    double t = (ball.area - DECEL_START_AREA_PX) / (BALL_CLOSE_AREA_PX - DECEL_START_AREA_PX);
                    t = std::clamp(t, 0.0, 1.0);
                    target_vx = MAX_FORWARD_SPEED - (float)t * (MAX_FORWARD_SPEED - MIN_APPROACH_SPEED);
                }
                // No sideways movement while approaching, only forward and turning
                target_vy = 0.0f;
                // If robot identifies ball is large for consecutive frames, 
                // that means we are close enough to start rotation/ALIGN
                if (approach_close_confirm.confirm(ball.area >= BALL_CLOSE_AREA_PX, APPROACH_CLOSE_CONFIRM_FRAMES)) {
                    // Switch to ALIGN/ORBITINg
                    phase = Phase::ALIGN;
                    // Print message confirming switch to ALIGN
                    std::cout << "[fetch] close to ball -> ALIGN\n";
                }
            }
            // DEBUG print for current phase + movement commands
            std::cout << "[fetch] phase=" << (int)phase << " target=(" << target_vx << "," << target_vy << "," << target_wz << ") ball_area=" << ball.area << "\n";
        }
        // Only run if currently in ALIGN phase, if ball becomes missing again go back to SEARCH
        else if (phase == Phase::ALIGN) {
            if (!ball.found) {
                target_vx = 0.0f; target_vy = 0.0f; target_wz = 0.0f;
                if (lost_ball_align_confirm.confirm(true, LOST_BALL_CONFIRM_FRAMES)) {
                    phase = Phase::SEARCH;
                    target_wz = SEARCH_SPIN_RATE;
                }
            // Ball IS visible in Frame
            } else {
                // Ball is not lost in this frame
                lost_ball_align_confirm.confirm(false, LOST_BALL_CONFIRM_FRAMES);
                // How off center the ball is
                double ball_offset = (ball.cx - CENTER_X) / (CAM_WIDTH / 2.0);
                // Turn toward ball based on the offset
                target_wz = std::clamp((float)(TURN_SIGN * BALL_TURN_GAIN * ball_offset), -1.0f, 1.0f);
                // No forward, only strafing/turning
                target_vx = 0.0f;
                // debug print, balss position, center ref., offset, turn command used
                std::cout << "[align-rotate-debug] ball.cx=" << ball.cx << " CENTER_X=" << CENTER_X
                            << " ball_offset=" << ball_offset << " target_wz=" << target_wz << "\n";
                // If the marker is not visible this frame
                if (!person.found) {
                    // strafe sideways at constantly speed to search for marker while circling the ball
                    target_vy = STRAFE_SIGN * CIRCLE_STRAFE_SPEED;
                    // ARE we aligned with ball and marker set to NO
                    align_coarse_confirm.confirm(false, ALIGN_COARSE_CONFIRM_FRAMES);

                // if the marker IS visible in frame
                } else {
                    // How far off center marker is
                    double person_err = person.cx - CENTER_X;
                    // How far off center ball is
                    double ball_err = ball.cx - CENTER_X;
                    // Debugging
                    std::cout << "[fetch] ALIGN ball_err=" << ball_err << " person_err=" << person_err << "\n";
                    // True if both ball and marker are centered
                    bool coarse_ok = std::abs(person_err) < COARSE_TOLERANCE_PX && std::abs(ball_err) < COARSE_TOLERANCE_PX;
                    // Count frame as roughly aligned
                    if (align_coarse_confirm.confirm(coarse_ok, ALIGN_COARSE_CONFIRM_FRAMES)) {
                        // Move to tighter FINE_ALIGN
                        phase = Phase::FINE_ALIGN;
                        // No strafing
                        target_vy = 0.0f;
                        // Print for mode change
                        std::cout << "[fetch] -> FINE_ALIGN\n";
                    // If not aligned enough yet...
                    } else {
                        // Sideways correction
                        double strafe = STRAFE_SIGN * STRAFE_GAIN * person_err;
                        // Apply sideways correction
                        target_vy = std::clamp((float)strafe, -CIRCLE_STRAFE_SPEED, CIRCLE_STRAFE_SPEED);
                    }
                }
            }
        }
        // If FINE ALIGN running but ball is not visible in frame
        else if (phase == Phase::FINE_ALIGN) {
            if (!ball.found) {
                target_vx = 0.0f; target_vy = 0.0f; target_wz = 0.0f;
                if (lost_ball_finealign_confirm.confirm(true, LOST_BALL_CONFIRM_FRAMES)) {
                    phase = Phase::SEARCH;
                    target_wz = SEARCH_SPIN_RATE;
                }

            // If ball is still visible
            } else {
                // is ball lost counter set to 0
                lost_ball_finealign_confirm.confirm(false, LOST_BALL_CONFIRM_FRAMES);
                // Readjust to center ball
                double ball_offset = (ball.cx - CENTER_X) / (CAM_WIDTH / 2.0);
                target_wz = std::clamp((float)(TURN_SIGN * BALL_TURN_GAIN * ball_offset), -1.0f, 1.0f);
                // Slowly creep forward toward the ball while adjusting
                target_vx = FINE_ALIGN_CREEP_SPEED;
                // If marker is not visible
                if (!person.found) {
                    // strafe sideways
                    target_vy = STRAFE_SIGN * CIRCLE_STRAFE_SPEED * 0.5f;
                    // tell preciesly aligned counter currently not aligned
                    finealign_fine_confirm.confirm(false, FINEALIGN_CONFIRM_FRAMES);

                // Marker is visible in frame
                } else {
                    // How far off center marker is
                    double person_err = person.cx - CENTER_X;
                    // how far off center ball is
                    double ball_err = ball.cx - CENTER_X;
                    // Debug, print offsets
                    std::cout << "[fetch] FINE_ALIGN ball_err=" << ball_err << " person_err=" << person_err << "\n";

                    // True if both ball and person are aligned correctly
                    bool fine_ok = std::abs(person_err) < FINE_TOLERANCE_PX && std::abs(ball_err) < FINE_TOLERANCE_PX;
                    // Counter frame as percisely aligned
                    if (finealign_fine_confirm.confirm(fine_ok, FINEALIGN_CONFIRM_FRAMES)) {
                        // Final charge to get ball to marker
                        phase = Phase::FINAL_PUSH;
                        final_push_counter = 0;
                        // no more strafing
                        target_vy = 0.0f;
                        // print phase change
                        std::cout << "[fetch] precisely aligned -> FINAL_PUSH\n";

                    // Ball and marker are NOT precisely aligned
                    } else {
                        // sideways adjustment to how far marker is + apply the adjustment with fine alignment movement speed
                        double strafe = STRAFE_SIGN * STRAFE_GAIN * person_err;
                        target_vy = std::clamp((float)strafe, -CIRCLE_STRAFE_SPEED * 0.5f, CIRCLE_STRAFE_SPEED * 0.5f);
                    }
                }
            }
        }
        // Only run while in WALK THROUGH
        else if (phase == Phase::WALK_THROUGH) {
            // If ball is visible in frame
            if (ball.found) {
                // Turn toward the ball
                double offset = (ball.cx - CENTER_X) / (CAM_WIDTH / 2.0);
                target_wz = std::clamp((float)(TURN_SIGN * TURN_GAIN * offset), -1.0f, 1.0f);
            }
            // Walk straight forward through the ball
            target_vx = MAX_FORWARD_SPEED; target_vy = 0.0f;
            // print marker's current size, and last known size
            std::cout << "[fetch] WALK_THROUGH person_area=" << person.area
                       << " last_person_area=" << last_person_area << "\n";
            // If marker is found, remember marker size, used later in case marker is out of frame
            if (person.found) {
                last_person_area = person.area;
            }
            // True if marker is visible and marker looks close enough to count as arrived
            bool arrived_visible = person.found && person.area > WALKTHROUGH_ARRIVED_AREA_PX;
            // Handles case where robot is too close to see the Marker
            bool arrived_via_fov_loss = !person.found && last_person_area > WALKTHROUGH_ARRIVED_AREA_PX;
            
            // Count frame toward "Arrived" counter if either condition above is true
            if (walkthrough_done_confirm.confirm(arrived_visible || arrived_via_fov_loss, WALKTHROUGH_DONE_CONFIRM_FRAMES)) {
                // State change to Final Push
                phase = Phase::FINAL_PUSH;
                final_push_counter = 0;
                // Debug, print which condition triggered Final Push
                std::cout << "[fetch] close enough (visible=" << arrived_visible
                          << " fov_loss=" << arrived_via_fov_loss << ") -> FINAL_PUSH\n";
            }
        }
        // Only run if in FINAL PUSH
        else if (phase == Phase::FINAL_PUSH) {
            // Run straightforward blind
            target_vx = FINAL_PUSH_SPEED; target_vy = 0.0f; target_wz = 0.0f;
            // Add step to final push counter
            final_push_counter++;
            // Debug, print step counter
            std::cout << "[fetch] FINAL_PUSH step " << final_push_counter << "/" << FINAL_PUSH_STEPS << " vx=" << target_vx << "\n";
            // If taken enough steps...
            if (final_push_counter >= FINAL_PUSH_STEPS) {
                // Stop the Go2
                target_vx = 0.0f; target_vy = 0.0f; target_wz = 0.0f;
                phase = Phase::DONE;
                std::cout << "[fetch] final push complete -> DONE (no re-throw on hardware; stopping)\n";
            }
        }
        // Debug, print current phase and all movement commands, runs every single loop iteration, regardless of phase
        std::cout << "[align-debug] phase=" << (int)phase << " vx=" << target_vx << " vy=" << target_vy << " wz=" << target_wz << "\n";
        // Send these movement commands to go2
        isaaclab::autopilot::set(target_vx, target_vy, target_wz);
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