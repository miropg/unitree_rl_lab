// Dataset capture for ball detector training.
// Video only -- never touches lowcmd, no FSM, no policy, no motor commands.
// Safe to run while driving the robot with the Unitree remote controller.
//
// Usage: ./capture_ball eth0 [output_dir]

#include <unitree/robot/go2/video/video_client.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <iostream>
#include <sys/stat.h>

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "usage: " << argv[0] << " <network_interface> [output_dir]\n";
        return 1;
    }
    std::string iface = argv[1];
    std::string outdir = (argc > 2) ? argv[2] : "/home/miro/ball_dataset";
    mkdir(outdir.c_str(), 0775);

    unitree::robot::ChannelFactory::Instance()->Init(0, iface);
    unitree::robot::go2::VideoClient video_client;
    video_client.SetTimeout(1.0f);
    video_client.Init();

    std::cout << "[capture] saving to " << outdir << "\n";
    std::cout << "[capture] drive the robot with the remote. Ctrl+C to stop.\n";

    const int SAVE_INTERVAL_MS = 500;   // 2 frames/sec
    int saved = 0, failed = 0;
    auto run_id = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(SAVE_INTERVAL_MS));

        std::vector<uint8_t> jpeg_bytes;
        int32_t ret = video_client.GetImageSample(jpeg_bytes);
        if (ret != 0 || jpeg_bytes.empty()) {
            if (++failed % 10 == 0)
                std::cout << "[capture] frame grab failing, ret=" << ret << "\n";
            continue;
        }
        cv::Mat frame = cv::imdecode(jpeg_bytes, cv::IMREAD_COLOR);
        if (frame.empty()) continue;

        char fn[256];
        snprintf(fn, sizeof(fn), "%s/ball_%ld_%05d.jpg",
                 outdir.c_str(), (long)run_id, saved);
        cv::imwrite(fn, frame);
        saved++;

        cv::Mat preview;
        cv::resize(frame, preview, cv::Size(960, 540));
        cv::putText(preview, "saved: " + std::to_string(saved),
                    cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cv::Scalar(0, 255, 0), 2);
        cv::imshow("capture - drive the robot around the ball", preview);
        cv::waitKey(1);

        if (saved % 20 == 0)
            std::cout << "[capture] " << saved << " frames\n";
    }
    return 0;
}
