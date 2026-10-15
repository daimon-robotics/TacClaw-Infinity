#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <stdexcept>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace daimon_aik {

enum class RobotVariant {
  TjFx,
  TjFxLite,
};

inline std::string ToString(const RobotVariant variant) {
  switch (variant) {
    case RobotVariant::TjFx:
      return "tj_fx";
    case RobotVariant::TjFxLite:
      return "tj_fx_lite";
  }
  return "tj_fx";
}

inline RobotVariant RobotVariantFromString(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (value == "tj_fx" || value == "full") {
    return RobotVariant::TjFx;
  }
  if (value == "tj_fx_lite" || value == "lite") {
    return RobotVariant::TjFxLite;
  }
  throw std::invalid_argument("Unsupported robot variant: " + value);
}

class ChainKinematics {
public:
  struct JointLimit {
    double lower = 0.0;
    double upper = 0.0;
  };

  struct JointSpec {
    Eigen::Vector3d xyz_m = Eigen::Vector3d::Zero();
    Eigen::Vector3d rpy_rad = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Rot = Eigen::Matrix3d::Identity();
    Eigen::Isometry3d TF = Eigen::Isometry3d::Identity();
    Eigen::Vector3d axis = Eigen::Vector3d::UnitZ();
    JointLimit limit;
    double theta0_rad = 0.0;
    JointLimit theta_limit;
  };

  struct AnalyticDhLink {
    double a_m = 0.0;
    double alpha_rad = 0.0;
    double d_m = 0.0;
    double theta0_rad = 0.0;
  };

  struct ArmGeometryConfig {
    JointSpec base{};
    std::array<JointSpec, 7> joints{};
    JointSpec ee{};
    std::array<AnalyticDhLink, 7> analytic_dh{};
    Eigen::Isometry3d T_base_dh0 = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d T_dh7_ee = Eigen::Isometry3d::Identity();
  };

  ChainKinematics() = default;

  ChainKinematics(const ArmGeometryConfig& left, const ArmGeometryConfig& right)
      : leftArmBase_(left.base),
        leftArmJoints_(left.joints),
        leftArmEe_(left.ee),
        rightArmBase_(right.base),
        rightArmJoints_(right.joints),
        rightArmEe_(right.ee) {}

  const JointSpec& getLeftArmBase() const { return leftArmBase_; }
  const std::array<JointSpec, 7>& getLeftArmJoints() const { return leftArmJoints_; }
  const JointSpec& getLeftArmEe() const { return leftArmEe_; }

  const JointSpec& getRightArmBase() const { return rightArmBase_; }
  const std::array<JointSpec, 7>& getRightArmJoints() const { return rightArmJoints_; }
  const JointSpec& getRightArmEe() const { return rightArmEe_; }

private:
  JointSpec leftArmBase_{};
  std::array<JointSpec, 7> leftArmJoints_{};
  JointSpec leftArmEe_{};

  JointSpec rightArmBase_{};
  std::array<JointSpec, 7> rightArmJoints_{};
  JointSpec rightArmEe_{};
};

struct JointLimitsConfig {
  struct JointLimit {
    double lower_rad = 0.0;
    double upper_rad = 0.0;
  };

  std::array<JointLimit, 7> q_limits_rad{};
  std::array<JointLimit, 7> dq_limits_rad_s{};
  std::array<JointLimit, 7> ddq_limits_rad_s2{};
  std::array<JointLimit, 7> dddq_limits_rad_s3{};
};

struct Dm25Model {
  struct DhLink {
    double a_m = 0.0;
    double alpha_rad = 0.0;
    double d_m = 0.0;
    double theta0_rad = 0.0;
  };

  std::array<DhLink, 7> dh{};
  std::array<double, 7> dh_alpha_cos{};
  std::array<double, 7> dh_alpha_sin{};
  JointLimitsConfig joint_limits{};

  Eigen::Isometry3d T_base_dh0 = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d T_dh7_ee = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d T_dh0_base = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d T_ee_dh7 = Eigen::Isometry3d::Identity();
};

} // namespace daimon_aik
