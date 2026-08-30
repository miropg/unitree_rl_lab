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

// CONFIRMATION CAMERA FRAME PARAMETERS                    * POSSIBLE ERRORS *
const int STARTUP_CONFIRM_FRAMES = 5;
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

// STARTUP SHORTCUT: if ball+vest are visible skip SEARCH/APPROACH_BALL
bool ENABLE_STARTUP_SHORTCUT = false;

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
const float SEARCH_SPIN_RATE = 0.0f;        // If ball is not in frame, do not spin searching for it (ethernet cord safety)

const double BALL_CLOSE_AREA_PX = 180000.0;
const double DECEL_START_AREA_PX = BALL_CLOSE_AREA_PX * 0.4; // start slowing down well before the close threshold
const float MIN_APPROACH_SPEED = 0.12f; // don't fully stop while still approaching, just slow down
const double COARSE_TOLERANCE_PX = 120.0;
const double FINE_TOLERANCE_PX = 30.0;
const float FINE_ALIGN_CREEP_SPEED = 0.08f;
const float BALL_TURN_GAIN = 3.0f;
const float CIRCLE_STRAFE_SPEED = 0.208f;
const float STRAFE_SIGN = 1.0f;
const float STRAFE_GAIN = 0.008f;

const int CAM_WIDTH = 1920;
const int CAM_HEIGHT = 1080;
const int CENTER_X = CAM_WIDTH / 2;

const double VISION_HZ = 2.0;
const double ALIGN_VISION_HZ = 6.0;

BlobResult find_vest(const cv::Mat& frame, cv::aruco::ArucoDetector& detector)
{
    BlobResult result;
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;

    detector.detectMarkers(frame, corners, ids);

    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] != VEST_MARKER_ID) continue;

        cv::Point2f center(0, 0);
        for (const auto& pt : corners[i]) center += pt;
        center *= 0.25f;

        cv::Rect box = cv::boundingRect(corners[i]);
        result.found = true;
        result.cx = center.x;
        result.cy = center.y;
        result.area = (double)box.width * box.height;
        break;
    }

    return result;
}

const int YOLO_INPUT_SIZE = 640;
const float BALL_CONF_THRESHOLD = 0.2f;
const int SPORTS_BALL_CLASS_ID = 32;
const std::string YOLO_MODEL_PATH = "/home/miro/unitree_rl_lab/deploy/robots/go2/yolov8s.onnx";

BlobResult find_ball(cv::dnn::Net& net, const cv::Mat& frame)
{
    BlobResult result;

    // Letterbox: resize preserving aspect ratio, pad remainder with gray (114,114,114),
    // matching Ultralytics' own preprocessing. A naive stretch-resize distorts round
    // objects into ovals and hurts detection, especially for smaller/distant objects.
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

    if (boxes.empty()) {
        std::cout << "[ball-debug] no candidate boxes above conf=" << BALL_CONF_THRESHOLD
                   << " (max confidence seen this frame: " << max_conf_seen << ")\n";
        return result;
    }

    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, confidences, BALL_CONF_THRESHOLD, 0.45f, nms_indices);
    if (nms_indices.empty()) {
        std::cout << "[ball-debug] " << boxes.size() << " candidate(s) found but NMS eliminated all\n";
        return result;
    }
    std::cout << "[ball-debug] " << boxes.size() << " candidate(s), " << nms_indices.size() << " survived NMS\n";

    int best_idx = nms_indices[0];
    float best_conf = confidences[best_idx];
    for (int idx : nms_indices) {
        if (confidences[idx] > best_conf) { best_conf = confidences[idx]; best_idx = idx; }
    }

    cv::Rect best_box = boxes[best_idx];
    result.found = true;
    result.cx = best_box.x + best_box.width / 2.0;
    result.cy = best_box.y + best_box.height / 2.0;
    result.area = (double)best_box.width * best_box.height;
    return result;
}

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

int main(int argc, char** argv)
{
    auto vm = param::helper(argc, argv);
    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     Go2 Fetch Controller \n";

    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());
    init_fsm_state();

    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "[fetch] Waiting 2s in Passive before standing up...\n";
    sleep(2);
    std::cout << "[fetch] Forcing FixStand...\n";
    fsm->forceState(2);
    sleep(4);
    std::cout << "[fetch] Forcing Velocity (policy control)...\n";
    fsm->forceState(3);
    sleep(1);

    unitree::robot::go2::VideoClient video_client;
    video_client.Init();

    cv::dnn::Net ball_net = cv::dnn::readNetFromONNX(YOLO_MODEL_PATH);
    ball_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    ball_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    cv::aruco::Dictionary aruco_dict = cv::aruco::getPredefinedDictionary(ARUCO_DICT_TYPE);
    cv::aruco::DetectorParameters aruco_params = cv::aruco::DetectorParameters();
    cv::aruco::ArucoDetector aruco_detector(aruco_dict, aruco_params);

    enum class Phase { SEARCH, APPROACH_BALL, ALIGN, FINE_ALIGN, WALK_THROUGH, FINAL_PUSH, DONE };
    Phase phase = Phase::SEARCH;
    bool startup_shortcut_used = false;
    FrameConfirm startup_confirm;
    FrameConfirm lost_ball_search_confirm;
    FrameConfirm lost_ball_align_confirm;
    FrameConfirm lost_ball_finealign_confirm;
    FrameConfirm approach_close_confirm;
    FrameConfirm align_coarse_confirm;
    FrameConfirm finealign_fine_confirm;
    FrameConfirm walkthrough_done_confirm;
    double last_person_area = 0.0;
    int final_push_counter = 0;

    float target_vx = 0.0f, target_vy = 0.0f, target_wz = SEARCH_SPIN_RATE;
    isaaclab::autopilot::set(target_vx, target_vy, target_wz);

    auto vision_period = std::chrono::milliseconds((int)(1000.0 / VISION_HZ));
    auto align_period = std::chrono::milliseconds((int)(1000.0 / ALIGN_VISION_HZ));

    std::cout << "[fetch] Starting fetch loop. Ctrl+C to stop.\n";

    while (phase != Phase::DONE)
    {
        auto period = (phase == Phase::ALIGN || phase == Phase::FINE_ALIGN) ? align_period : vision_period;
        std::this_thread::sleep_for(period);

        std::vector<uint8_t> jpeg_bytes;
        int32_t ret = video_client.GetImageSample(jpeg_bytes);
        if (ret != 0 || jpeg_bytes.empty()) {
            std::cout << "[warning] camera frame grab failed, ret=" << ret << "\n";
            continue;
        }
        cv::Mat frame = cv::imdecode(jpeg_bytes, cv::IMREAD_COLOR);
        if (frame.empty()) continue;
        std::cout << "[debug] frame size: " << frame.cols << "x" << frame.rows << "\n";

        cv::Mat hsv;

        BlobResult ball = find_ball(ball_net, frame);
        BlobResult person = find_vest(frame, aruco_detector);

        cv::Mat display = frame.clone();
        cv::line(display, cv::Point(CENTER_X, 0), cv::Point(CENTER_X, CAM_HEIGHT), cv::Scalar(0,0,0), 1);
        if (ball.found) cv::drawMarker(display, cv::Point((int)ball.cx, (int)ball.cy), cv::Scalar(0,255,0), cv::MARKER_CROSS, 20, 2);
        if (person.found) cv::drawMarker(display, cv::Point((int)person.cx, (int)person.cy), cv::Scalar(0,140,255), cv::MARKER_CROSS, 20, 2);
        cv::imshow("Go2 Fetch - live view", display);
        cv::waitKey(1);

        // --- One-time startup shortcut ---
        if (ENABLE_STARTUP_SHORTCUT && !startup_shortcut_used && phase == Phase::SEARCH) {
            bool ready_walk = ball.found && person.found && ball.area >= BALL_CLOSE_AREA_PX &&
                               std::abs(ball.cx - CENTER_X) < FINE_TOLERANCE_PX &&
                               std::abs(person.cx - CENTER_X) < FINE_TOLERANCE_PX;
            bool ready_fine = ball.found && person.found && ball.area >= BALL_CLOSE_AREA_PX &&
                               std::abs(ball.cx - CENTER_X) < COARSE_TOLERANCE_PX &&
                               std::abs(person.cx - CENTER_X) < COARSE_TOLERANCE_PX;
            bool ready_align = ball.found && ball.area >= BALL_CLOSE_AREA_PX;
            bool any_ready = ready_walk || ready_fine || ready_align;

            if (startup_confirm.confirm(any_ready, STARTUP_CONFIRM_FRAMES)) {
                startup_shortcut_used = true;
                if (ready_walk) {
                    phase = Phase::WALK_THROUGH;
                    std::cout << "[fetch] startup shortcut: already aligned -> WALK_THROUGH\n";
                } else if (ready_fine) {
                    phase = Phase::FINE_ALIGN;
                    std::cout << "[fetch] startup shortcut: coarse aligned -> FINE_ALIGN\n";
                } else {
                    phase = Phase::ALIGN;
                    std::cout << "[fetch] startup shortcut: ball close -> ALIGN\n";
                }
            } else if (!any_ready) {
                startup_shortcut_used = true;
            }
        }

        if (phase == Phase::SEARCH || phase == Phase::APPROACH_BALL) {
            if (!ball.found) {
                approach_close_confirm.confirm(false, APPROACH_CLOSE_CONFIRM_FRAMES);
                if (lost_ball_search_confirm.confirm(true, LOST_BALL_CONFIRM_FRAMES)) {
                    phase = Phase::SEARCH;
                    target_wz = SEARCH_SPIN_RATE;
                }
            } else {
                lost_ball_search_confirm.confirm(false, LOST_BALL_CONFIRM_FRAMES);
                phase = Phase::APPROACH_BALL;
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
                    phase = Phase::ALIGN;
                    std::cout << "[fetch] close to ball -> ALIGN\n";
                }
            }
            std::cout << "[fetch] phase=" << (int)phase << " target=(" << target_vx << "," << target_vy << "," << target_wz << ") ball_area=" << ball.area << "\n";
        }
        else if (phase == Phase::ALIGN) {
            if (!ball.found) {
                if (lost_ball_align_confirm.confirm(true, LOST_BALL_CONFIRM_FRAMES)) {
                    phase = Phase::SEARCH;
                    target_wz = SEARCH_SPIN_RATE;
                }
            } else {
                lost_ball_align_confirm.confirm(false, LOST_BALL_CONFIRM_FRAMES);
                double ball_offset = (ball.cx - CENTER_X) / (CAM_WIDTH / 2.0);
                target_wz = std::clamp((float)(TURN_SIGN * BALL_TURN_GAIN * ball_offset), -1.0f, 1.0f);
                target_vx = 0.0f;
                if (!person.found) {
                    target_vy = STRAFE_SIGN * CIRCLE_STRAFE_SPEED;
                    align_coarse_confirm.confirm(false, ALIGN_COARSE_CONFIRM_FRAMES);
                } else {
                    double person_err = person.cx - CENTER_X;
                    double ball_err = ball.cx - CENTER_X;
                    std::cout << "[fetch] ALIGN ball_err=" << ball_err << " person_err=" << person_err << "\n";
                    bool coarse_ok = std::abs(person_err) < COARSE_TOLERANCE_PX && std::abs(ball_err) < COARSE_TOLERANCE_PX;
                    if (align_coarse_confirm.confirm(coarse_ok, ALIGN_COARSE_CONFIRM_FRAMES)) {
                        phase = Phase::FINE_ALIGN;
                        target_vy = 0.0f;
                        std::cout << "[fetch] -> FINE_ALIGN\n";
                    } else {
                        double strafe = STRAFE_SIGN * STRAFE_GAIN * person_err;
                        target_vy = std::clamp((float)strafe, -CIRCLE_STRAFE_SPEED, CIRCLE_STRAFE_SPEED);
                    }
                }
            }
        }
        else if (phase == Phase::FINE_ALIGN) {
            if (!ball.found) {
                if (lost_ball_finealign_confirm.confirm(true, LOST_BALL_CONFIRM_FRAMES)) {
                    phase = Phase::SEARCH;
                    target_wz = SEARCH_SPIN_RATE;
                }
            } else {
                lost_ball_finealign_confirm.confirm(false, LOST_BALL_CONFIRM_FRAMES);
                double ball_offset = (ball.cx - CENTER_X) / (CAM_WIDTH / 2.0);
                target_wz = std::clamp((float)(TURN_SIGN * BALL_TURN_GAIN * ball_offset), -1.0f, 1.0f);
                target_vx = FINE_ALIGN_CREEP_SPEED;
                if (!person.found) {
                    target_vy = STRAFE_SIGN * CIRCLE_STRAFE_SPEED * 0.5f;
                    finealign_fine_confirm.confirm(false, FINEALIGN_CONFIRM_FRAMES);
                } else {
                    double person_err = person.cx - CENTER_X;
                    double ball_err = ball.cx - CENTER_X;
                    std::cout << "[fetch] FINE_ALIGN ball_err=" << ball_err << " person_err=" << person_err << "\n";
                    bool fine_ok = std::abs(person_err) < FINE_TOLERANCE_PX && std::abs(ball_err) < FINE_TOLERANCE_PX;
                    if (finealign_fine_confirm.confirm(fine_ok, FINEALIGN_CONFIRM_FRAMES)) {
                        phase = Phase::FINAL_PUSH;
                        final_push_counter = 0;
                        target_vy = 0.0f;
                        std::cout << "[fetch] precisely aligned -> FINAL_PUSH\n";
                    } else {
                        double strafe = STRAFE_SIGN * STRAFE_GAIN * person_err;
                        target_vy = std::clamp((float)strafe, -CIRCLE_STRAFE_SPEED * 0.5f, CIRCLE_STRAFE_SPEED * 0.5f);
                    }
                }
            }
        }
        else if (phase == Phase::WALK_THROUGH) {
            if (ball.found) {
                double offset = (ball.cx - CENTER_X) / (CAM_WIDTH / 2.0);
                target_wz = std::clamp((float)(TURN_SIGN * TURN_GAIN * offset), -1.0f, 1.0f);
            }
            target_vx = MAX_FORWARD_SPEED; target_vy = 0.0f;
            std::cout << "[fetch] WALK_THROUGH person_area=" << person.area
                       << " last_person_area=" << last_person_area << "\n";

            if (person.found) {
                last_person_area = person.area;
            }

            bool arrived_visible = person.found && person.area > WALKTHROUGH_ARRIVED_AREA_PX;
            bool arrived_via_fov_loss = !person.found && last_person_area > WALKTHROUGH_ARRIVED_AREA_PX;

            if (walkthrough_done_confirm.confirm(arrived_visible || arrived_via_fov_loss, WALKTHROUGH_DONE_CONFIRM_FRAMES)) {
                phase = Phase::FINAL_PUSH;
                final_push_counter = 0;
                std::cout << "[fetch] close enough (visible=" << arrived_visible
                          << " fov_loss=" << arrived_via_fov_loss << ") -> FINAL_PUSH\n";
            }
        }
        else if (phase == Phase::FINAL_PUSH) {
            target_vx = FINAL_PUSH_SPEED; target_vy = 0.0f; target_wz = 0.0f;
            final_push_counter++;
            std::cout << "[fetch] FINAL_PUSH step " << final_push_counter << "/" << FINAL_PUSH_STEPS << " vx=" << target_vx << "\n";
            if (final_push_counter >= FINAL_PUSH_STEPS) {
                target_vx = 0.0f; target_vy = 0.0f; target_wz = 0.0f;
                phase = Phase::DONE;
                std::cout << "[fetch] final push complete -> DONE (no re-throw on hardware; stopping)\n";
            }
        }

        std::cout << "[align-debug] phase=" << (int)phase << " vx=" << target_vx << " vy=" << target_vy << " wz=" << target_wz << "\n";
        isaaclab::autopilot::set(target_vx, target_vy, target_wz);
    }

    isaaclab::autopilot::stop();
    cv::destroyAllWindows();
    std::cout << "[fetch] Done. Holding position.\n";
    while (true) { sleep(1); }
    return 0;
}
