cmake_minimum_required(VERSION 2.8.3)
project(tmc_control_msgs)

find_package(catkin REQUIRED COMPONENTS std_msgs message_generation actionlib_msgs)
add_message_files(
    FILES
    ServoParam.msg
    ServoState.msg
    ExxxDriveMode.msg
    JointExxxDriveMode.msg
	OverloadJointState.msg
)

add_service_files(
  FILES
  ChangeExxxDriveMode.srv
  InitActuatorEncorder.srv
  ReadParameters.srv
  WriteParameters.srv
  SetBumperMode.srv
)

add_action_files(
  FILES
  GripperApplyEffort.action
)

generate_messages(
    DEPENDENCIES
    std_msgs
    actionlib_msgs
)
catkin_package(
  CATKIN_DEPENDS std_msgs message_runtime actionlib_msgs
)

