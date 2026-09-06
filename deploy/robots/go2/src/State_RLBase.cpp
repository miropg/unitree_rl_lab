#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <fstream>
#include <chrono>

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    // DIAGNOSTIC: throttled joint-command log to isolate a repeatable front-right leg
    // stall/limp. Prints every 5th tick to catch brief stalls without heavy I/O overhead.
    // Runs under BOTH main_fetch and plain joystick-driven main.cpp (both call this
    // function) -- useful for direct comparison.
    // Indices 1, 5, 9 = front-right hip/thigh/calf, inferred from deploy.yaml's
    // joint_ids_map against Unitree's standard 0-11 motor order (0-2 FR, 3-5 FL,
    // 6-8 RR, 9-11 RL). UNVERIFIED against the SDK header directly -- sanity check
    // these three during ordinary walking before trusting them exclusively.
    // --- joint debug: commanded vs measured -> /tmp/joint_debug.csv ---
    {
        static std::ofstream jd("/tmp/joint_debug.csv");
        static bool jd_init = false;
        static int  jd_flush = 0;
        if (!jd_init) {
            jd << "t_ms,"
               << "cmd_FR_hip,cmd_FR_thigh,cmd_FR_calf,"
               << "meas_FR_hip,meas_FR_thigh,meas_FR_calf,"
               << "cmd_FL_hip,cmd_FL_thigh,cmd_FL_calf,"
               << "meas_FL_hip,meas_FL_thigh,meas_FL_calf\n";
            jd_init = true;
        }
        const auto & jp = env->robot->data.joint_pos;
        jd << std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now().time_since_epoch()).count()
           << "," << action[1] << "," << action[5] << "," << action[9]
           << "," << jp[1]     << "," << jp[5]     << "," << jp[9]
           << "," << action[0] << "," << action[4] << "," << action[8]
           << "," << jp[0]     << "," << jp[4]     << "," << jp[8]
           << "\n";
        if (++jd_flush % 25 == 0) jd.flush();
    }

    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}