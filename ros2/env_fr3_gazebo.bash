#!/bin/bash

# Source this file in each terminal used for GELLO teleoperation in Gazebo.
_gello_ros2_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)" || return 1
source "$_gello_ros2_dir/../.venv/bin/activate" || return 1
source "$_gello_ros2_dir/../../franka_ws/install/setup.bash" || return 1
source "$_gello_ros2_dir/install/local_setup.bash" || return 1
unset _gello_ros2_dir

export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-89}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR="$HOME/.ros/log"

export IGN_PARTITION="gello_fr3_gazebo_${ROS_DOMAIN_ID}"
export GZ_PARTITION="$IGN_PARTITION"

# Gazebo resolves package:// mesh URIs relative to the package's parent directory.
_gello_description_share="$(ros2 pkg prefix --share franka_description)" || return 1
_gello_bringup_share="$(ros2 pkg prefix --share franka_gazebo_bringup)" || return 1
export GZ_SIM_RESOURCE_PATH="${_gello_description_share%/*}:$_gello_bringup_share"
export IGN_GAZEBO_RESOURCE_PATH="$GZ_SIM_RESOURCE_PATH"
unset _gello_description_share _gello_bringup_share
