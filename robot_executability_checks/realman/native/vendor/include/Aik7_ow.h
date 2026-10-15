#pragma once

#include <array>
#include <vector>

#include <daimon_robot_model_types/RobotModelTypes.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace daimon_aik {

/// 解析逆解入口：返回给定肘部构型下的所有解析分支解（经限位过滤 + 去重）。
///
/// - 输入 `T_base_ee`: base_link->ee 的位姿（与 YAML meta.base_link / meta.ee_link 同一坐标系约定）
/// - 输入 `q7_rad`: URDF q-space 的第 7 轴角度（rad，冗余输入）
/// - 输入 `elbow_sign`: 肘部构型符号（>=0 → theta4 取正值；<0 → theta4 取负值）
/// - 输出：所有解的 q-space（URDF 关节角，rad），其中第 7 轴固定等于输入 `q7_rad`。
///         若目标不可达（|cos(theta4)| > 1），返回空 vector。
std::vector<std::array<double, 7>> SolveAik7_ow_left(
    const Eigen::Isometry3d& T_base_ee,
    double q7_rad,
    const Dm25Model& model,
    int elbow_sign = -1);

/// 右臂版解析逆解入口：与 SolveAik7_ow 相同，但使用右臂 URDF 限位过滤。
std::vector<std::array<double, 7>> SolveAik7_ow_right(
    const Eigen::Isometry3d& T_base_ee,
    double q7_rad,
    const Dm25Model& model,
    int elbow_sign = +1);

} // namespace daimon_aik
