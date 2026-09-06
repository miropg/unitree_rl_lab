#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "isaaclab/utils/autopilot.h"
std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;
void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}
int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);
    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     Go2 Controller \n";
    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());
    init_fsm_state();
    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    // --- Scripted autopilot sequence (no physical remote available) ---
    // State ids come from deploy/robots/go2/config/config.yaml:
    //   Passive = 1, FixStand = 2, Velocity = 3
    std::cout << "[autopilot] Waiting 2s in Passive before standing up...\n";
    sleep(2);

    std::cout << "[autopilot] Forcing FixStand...\n";
    fsm->forceState(2);
    sleep(4); // let the stand-up sequence finish

    std::cout << "[autopilot] Forcing Velocity (policy control)...\n";
    fsm->forceState(3);
    sleep(1); // let the policy settle into a stable stance

    const float WALK_DURATION_SEC = 50.0f;
    const float FORWARD_SPEED = 0.4f; // clamped by deploy.yaml's lin_vel_x range
    std::cout << "[autopilot] Walking forward for " << WALK_DURATION_SEC << "s...\n";
    isaaclab::autopilot::set(FORWARD_SPEED, 0.0f, 0.0f);
    sleep((unsigned int)WALK_DURATION_SEC);

    std::cout << "[autopilot] Stopping.\n";
    isaaclab::autopilot::stop();

    while (true)
    {
        sleep(1);
    }
    return 0;
}
