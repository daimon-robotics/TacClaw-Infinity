// Standard-DH representation of the current TJ_FX complete URDF.
//
// Derived from the fixed joints and Joint1..Joint7 axes in
// models/ros_packages/daimon_tj_fx_description/urdf/
// daimon_tj_fx_dual_arm_with_gripper.urdf.  It intentionally preserves the
// joint order and q-space names used by MuJoCo.  This lets the upstream Aik7
// closed-form equations operate on the same ee_link FK that MuJoCo renders.

#include "current_urdf_model.h"

#include <array>
#include <cmath>
#include <stdexcept>

namespace tianji_aik7_model {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kHalfPi = 1.57079632679489661923;

Eigen::Matrix3d rotationZ(const double angle) {
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  Eigen::Matrix3d rotation;
  rotation << c, -s, 0.0,
      s, c, 0.0,
      0.0, 0.0, 1.0;
  return rotation;
}

Eigen::Isometry3d makeTransform(
    const Eigen::Matrix3d& rotation,
    const Eigen::Vector3d& translation) {
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.linear() = rotation;
  transform.translation() = translation;
  return transform;
}

void finalizeModel(daimon_aik::Dm25Model* model) {
  model->T_dh0_base = model->T_base_dh0.inverse();
  model->T_ee_dh7 = model->T_dh7_ee.inverse();
  for (size_t index = 0; index < model->dh.size(); ++index) {
    model->dh_alpha_cos[index] = std::cos(model->dh[index].alpha_rad);
    model->dh_alpha_sin[index] = std::sin(model->dh[index].alpha_rad);
  }
}

}  // namespace

daimon_aik::Dm25Model CreateCurrentMuJoCoUrdfModel(const int side) {
  if (side != 0 && side != 1) {
    throw std::invalid_argument("side must be 0 (left) or 1 (right)");
  }

  // The parameterization is a standard-DH rewrite of the current URDF.  The
  // two 18 mm lateral terms are intentionally signed: that is what preserves
  // the URDF's Link3->Link4 and Link4->Link5 fixed transforms without a
  // hidden per-joint correction.
  constexpr std::array<double, 7> kA = {0.0, 0.0, -0.018, 0.018, 0.0, 0.0, 0.095};
  constexpr std::array<double, 7> kAlpha = {
      -kHalfPi, kHalfPi, -kHalfPi, kHalfPi, -kHalfPi, kHalfPi, 0.0};
  constexpr std::array<double, 7> kD = {0.0, 0.0, 0.287, 0.0, 0.314, 0.0, 0.0};
  constexpr std::array<double, 7> kTheta0 = {0.0, 0.0, 0.0, 0.0, 0.0, -kHalfPi, 0.0};
  constexpr std::array<std::array<double, 2>, 7> kLimits = {{
      {{-3.1067, 3.1067}},
      {{-2.0944, 2.0944}},
      {{-3.1067, 3.1067}},
      {{-2.5307, 1.0472}},
      {{-3.1067, 3.1067}},
      {{-1.0472, 1.0472}},
      {{-1.5708, 1.5708}},
  }};

  daimon_aik::Dm25Model model;
  for (size_t index = 0; index < model.dh.size(); ++index) {
    model.dh[index] = daimon_aik::Dm25Model::DhLink{
        kA[index], kAlpha[index], kD[index], kTheta0[index]};
    model.joint_limits.q_limits_rad[index] = {
        kLimits[index][0], kLimits[index][1]};
  }

  // At the analytical base frame, joint 1 is located 0.060 + 0.1745 m from
  // base_link.  The reflection-looking first column is a valid proper
  // rotation (det=+1), not a mirror of the robot model.
  Eigen::Matrix3d base_rotation;
  Eigen::Vector3d base_translation;
  if (side == 0) {
    base_rotation << -1.0, 0.0, 0.0,
        0.0, 0.0, 1.0,
        0.0, 1.0, 0.0;
    base_translation = Eigen::Vector3d(0.0, 0.2345, 0.0);
  } else {
    base_rotation << -1.0, 0.0, 0.0,
        0.0, 0.0, -1.0,
        0.0, -1.0, 0.0;
    base_translation = Eigen::Vector3d(0.0, -0.2345, 0.0);
  }
  model.T_base_dh0 = makeTransform(base_rotation, base_translation);
  model.T_dh7_ee = makeTransform(rotationZ(kHalfPi), Eigen::Vector3d::Zero());
  finalizeModel(&model);
  return model;
}

}  // namespace tianji_aik7_model
