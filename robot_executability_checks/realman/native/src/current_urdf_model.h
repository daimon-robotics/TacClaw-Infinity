#pragma once

#include <daimon_robot_model_types/RobotModelTypes.h>

namespace tianji_aik7_model {

// Build the standard-DH model derived from the exact joint chain of this
// repository's daimon_tj_fx_dual_arm_with_gripper.urdf.  The ROS workspace's
// original Dm25Model uses a different frame convention, so it must not be
// sent directly to MuJoCo with the same qpos values.
daimon_aik::Dm25Model CreateCurrentMuJoCoUrdfModel(int side);

}  // namespace tianji_aik7_model
