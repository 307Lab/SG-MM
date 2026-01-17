//
// Created by honerkam on 10/1/21.
//

#ifndef MODULATION_RL_ROBOT_FG125_H
#define MODULATION_RL_ROBOT_FG125_H

#include <actionlib/client/simple_action_client.h>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include <pybind11/pybind11.h>
#include <modulation_rl/robot_env.h>
// #include <controller_manager_msgs/SwitchController.h>

typedef actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction> TrajClientFg125;
typedef boost::shared_ptr<TrajClientFg125> Fg125ClientPtr;

class RobotFg125 : public RobotEnv {
private:
    Fg125ClientPtr arm_client_;
    // HuskyClientPtr torso_client_;
    Fg125ClientPtr gripper_client_;
    // ros::ServiceClient switch_controller_client_;
    // moveit::planning_interface::MoveGroupInterface* move_group_arm_torso_;
    control_msgs::FollowJointTrajectoryGoal arm_goal_;
    // control_msgs::FollowJointTrajectoryGoal torso_goal_;

    void setup();
    bool getArmSuccess() override;
    // void stop_controllers();
    // void start_controllers();

    tf::Transform calcDesiredBaseTf(const tf::Transform &base_tf,
                                    const tf::Vector3 &base_translation_relative,
                                    double base_rotation_relative,
                                    double dt) override;
    void calcDesiredBaseCommand(const tf::Transform &current_base_tf,
                                const tf::Transform &desired_base_tf,
                                double dt,
                                tf::Vector3 &base_translation_per_second,
                                double &base_rotation_per_second) override;

public:
  RobotFg125(uint32_t seed,
             const std::string &node_handle_name,
             bool init_controllers,
             const std::string &world_type,
             const RoboConf &robot_conf,
             double bioik_center_joints_weight,
             double bioik_avoid_joint_limits_weight,
             double bioik_regularization_weight,
             const std::string &bioik_regularization_type
    );
    ~RobotFg125() {
    }

    void openGripper(double position, bool wait_for_result) override;
    void closeGripper(double position, bool wait_for_result) override;
    void sendArmCommands(const std::vector<std::vector<double>> &joint_values, std::vector<double> &timestamps) override;
};


#endif  // MODULATION_RL_ROBOT_FG125_H
