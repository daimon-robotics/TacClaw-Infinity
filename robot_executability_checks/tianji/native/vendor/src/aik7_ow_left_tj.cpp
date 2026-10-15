#include "Aik7_ow.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

namespace daimon_aik {

namespace {

Eigen::Isometry3d dhTStandardCS(
    const double ct,
    const double st,
    const Dm25Model::DhLink& link,
    const double ca,
    const double sa) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear()(0, 0) = ct;
  T.linear()(0, 1) = -st * ca;
  T.linear()(0, 2) = st * sa;
  T.linear()(1, 0) = st;
  T.linear()(1, 1) = ct * ca;
  T.linear()(1, 2) = -ct * sa;
  T.linear()(2, 0) = 0.0;
  T.linear()(2, 1) = sa;
  T.linear()(2, 2) = ca;
  T.translation().x() = link.a_m * ct;
  T.translation().y() = link.a_m * st;
  T.translation().z() = link.d_m;
  return T;
}

Eigen::Isometry3d dhTStandard(
    const double theta,
    const Dm25Model::DhLink& link,
    const double ca,
    const double sa) {
  const double ct = std::cos(theta);
  const double st = std::sin(theta);
  return dhTStandardCS(ct, st, link, ca, sa);
}

Eigen::Matrix3d skew3(const Eigen::Vector3d& u) {
  Eigen::Matrix3d ux = Eigen::Matrix3d::Zero();
  ux(0, 1) = -u.z();
  ux(0, 2) = u.y();
  ux(1, 0) = u.z();
  ux(1, 2) = -u.x();
  ux(2, 0) = -u.y();
  ux(2, 1) = u.x();
  return ux;
}

double signNoZero(const double x) { return (x >= 0.0) ? 1.0 : -1.0; }

using JointLimit = JointLimitsConfig::JointLimit;

bool isWithinLimit(const double q, const JointLimit& lim, const double tol_rad) {
  if (!std::isfinite(q)) {
    return false;
  }
  return (lim.lower_rad - tol_rad) <= q && q <= (lim.upper_rad + tol_rad);
}

double wrapAngleIntoRange(const double q, const JointLimit& lim, const double tol_rad) {
  if (!std::isfinite(q)) {
    return q;
  }
  if (!(std::isfinite(lim.lower_rad) && std::isfinite(lim.upper_rad) && (lim.upper_rad > lim.lower_rad))) {
    return q;
  }

  constexpr double kTwoPi = 2.0 * M_PI;
  const double mid = 0.5 * (lim.lower_rad + lim.upper_rad);

  const long long n0 = std::llround((mid - q) / kTwoPi);

  double best = q;
  double best_abs = std::numeric_limits<double>::infinity();
  bool found = false;

  for (long long n = (n0 - 1); n <= (n0 + 1); ++n) {
    const double qc = q + static_cast<double>(n) * kTwoPi;
    if (!isWithinLimit(qc, lim, tol_rad)) {
      continue;
    }
    const double d = std::abs(qc - mid);
    if (d < best_abs) {
      best_abs = d;
      best = qc;
      found = true;
    }
  }

  if (!found) {
    return q;
  }

  // If we are only slightly outside due to numerical tolerance, snap to boundary.
  if (best < lim.lower_rad) {
    best = lim.lower_rad;
  }
  if (best > lim.upper_rad) {
    best = lim.upper_rad;
  }
  return best;
}

Eigen::Isometry3d fkBaseEeFromQ(const std::array<double, 7>& q, const Dm25Model& model) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  for (size_t i = 0; i < 7; ++i) {
    const double theta = q[i] + model.dh[i].theta0_rad;
    T = T * dhTStandard(theta, model.dh[i], model.dh_alpha_cos[i], model.dh_alpha_sin[i]);
  }
  return model.T_base_dh0 * T * model.T_dh7_ee;
}

double rotationDistanceRad(const Eigen::Matrix3d& R_a, const Eigen::Matrix3d& R_b) {
  const Eigen::Matrix3d R_err = R_a.transpose() * R_b;
  double c = 0.5 * (R_err.trace() - 1.0);
  if (c > 1.0) {
    c = 1.0;
  } else if (c < -1.0) {
    c = -1.0;
  }
  return std::acos(c);
}

bool passesForwardPoseValidation(
    const std::array<double, 7>& q,
    const Eigen::Isometry3d& T_target_base_ee,
    const Dm25Model& model,
    const double pos_tol_m,
    const double rot_tol_rad) {
  const Eigen::Isometry3d T_fk_base_ee = fkBaseEeFromQ(q, model);
  const double pos_err_m = (T_fk_base_ee.translation() - T_target_base_ee.translation()).norm();
  const double rot_err_rad = rotationDistanceRad(T_fk_base_ee.linear(), T_target_base_ee.linear());
  return (pos_err_m <= pos_tol_m) && (rot_err_rad <= rot_tol_rad);
}

// ----------------------------
// Elbow angle (theta4) – single solution controlled by elbow_sign
// ----------------------------

double wrapToPi(const double angle_rad) {
  return std::atan2(std::sin(angle_rad), std::cos(angle_rad));
}

/// TJ_FX 肘部几何相比旧 DM25 额外包含一段 a3≈18 mm 的横向偏置。
/// 在当前标准 DH 下，肩到腕向量在 frame3 中仍然可写成：
///   v3(theta4) = u*cos(theta4) + v*sin(theta4) + w
/// 因此 |v3| = |L_sw| 依然能化成单个线性三角方程 A*cos + B*sin + C = 0。
std::optional<double> computeTheta4(
    const Eigen::Vector3d& L_sw_0,
    const Dm25Model& model,
    int elbow_sign) {
  const Eigen::Isometry3d T_2_3 =
      dhTStandard(0.0, model.dh[2], model.dh_alpha_cos[2], model.dh_alpha_sin[2]);
  const Eigen::Vector3d L_23_3 = T_2_3.linear().transpose() * T_2_3.translation();

  const double a3 = model.dh[3].a_m;
  const double d4 = model.dh[3].d_m;
  const double d5 = model.dh[4].d_m;
  const double sa = model.dh_alpha_sin[3];
  const double ca = model.dh_alpha_cos[3];

  const Eigen::Vector3d u(a3, -d5 * sa, 0.0);
  const Eigen::Vector3d v(d5 * sa, a3, 0.0);
  const Eigen::Vector3d w = L_23_3 + Eigen::Vector3d(0.0, 0.0, d4 + d5 * ca);

  const double target_len_sq = L_sw_0.squaredNorm();
  const double A = 2.0 * u.dot(w);
  const double B = 2.0 * v.dot(w);
  const double C = u.squaredNorm() + w.squaredNorm() - target_len_sq;

  const double M = std::hypot(A, B);
  if (M <= 1e-12) {
    return std::nullopt;
  }

  double arg = -C / M;
  if (arg > 1.0 || arg < -1.0) {
    return std::nullopt;
  }
  arg = std::max(-1.0, std::min(1.0, arg));

  const double phi = std::atan2(B, A);
  const double gamma = std::acos(arg);

  const std::array<double, 2> roots = {
      wrapToPi(phi + gamma),
      wrapToPi(phi - gamma),
  };

  const double preferred_sign = (elbow_sign >= 0) ? 1.0 : -1.0;
  double best = roots[0];
  double best_score = -1e9;
  for (const double theta : roots) {
    double score = -std::abs(theta);
    if (theta * preferred_sign >= -1e-9) {
      score += 1000.0;
    }
    if (score > best_score) {
      best_score = score;
      best = theta;
    }
  }
  return best;
}

std::array<std::array<double, 2>, 2> theta12_0TwoBranches(
    const Eigen::Vector3d& L_sw_0,
    const Eigen::Vector3d& v3) {
  const double Lx = L_sw_0.x();
  const double Ly = L_sw_0.y();
  const double Lz = L_sw_0.z();
  const double vx = v3.x();
  const double vy = v3.y();
  const double vz = v3.z();

  const double r = std::hypot(vx, vy);
  const double phi = std::atan2(vx, vy);

  const double k2 = std::max(0.0, Lx * Lx + Ly * Ly - vz * vz);
  const double k_abs = std::sqrt(k2);

  const double s2 = std::max(0.0, r * r - Lz * Lz);
  const double alpha = std::atan2(std::sqrt(s2), -Lz);

  const double theta2_a = phi + alpha;
  const double theta2_b = phi - alpha;

  const auto theta1_from_A = [&](const double A) -> double {
    return std::atan2(-vz * Lx + A * Ly, A * Lx + vz * Ly);
  };

  return {{
      {theta1_from_A(-k_abs), theta2_a},
      {theta1_from_A(+k_abs), theta2_b},
  }};
}

std::array<double, 2> phiTwoBranches(
    const Eigen::Matrix3d& Aw,
    const Eigen::Matrix3d& Bw,
    const Eigen::Matrix3d& Cw,
    const double s7,
    const double c7) {
  // Solve Eq.(27) in the linear trig form:
  //   A*sin(phi) + B*cos(phi) + C = 0
  //
  // Original derivation used t7 = tan(theta7):
  //   A = Aw(2,1) + Aw(2,0)*t7, etc.
  // Multiply the whole equation by cos(theta7) to avoid tan() and division:
  //   A' = Aw(2,1)*c7 + Aw(2,0)*s7, etc.
  const double A = Aw(2, 1) * c7 + Aw(2, 0) * s7;
  const double B = Bw(2, 1) * c7 + Bw(2, 0) * s7;
  const double C = Cw(2, 1) * c7 + Cw(2, 0) * s7;

  const double R = std::hypot(A, B);

  const double delta = std::atan2(A, B); // A = R*sin(delta), B = R*cos(delta)
  double arg = -C / R;                   // cos(phi - delta) = -C/R
  if (arg > 1.0) {
    arg = 1.0;
  } else if (arg < -1.0) {
    arg = -1.0;
  }
  const double gamma = std::acos(arg);

  return {delta + gamma, delta - gamma};
}

std::array<std::array<double, 3>, 2> theta123TwoBranches(
    const Eigen::Matrix3d& As,
    const Eigen::Matrix3d& Bs,
    const Eigen::Matrix3d& Cs,
    const double phi) {
  const double sphi = std::sin(phi);
  const double cphi = std::cos(phi);
  const Eigen::Matrix3d R0_3 = As * sphi + Bs * cphi + Cs;

  const double r12 = R0_3(0, 1);
  const double r22 = R0_3(1, 1);
  const double r31 = R0_3(2, 0);
  const double r32 = R0_3(2, 1);
  const double r33 = R0_3(2, 2);


  const double theta2_p = std::acos(-r32);
  const double theta2_m = -theta2_p;

  const auto solve = [&](const double theta2) -> std::array<double, 3> {
    const double sgn2 = signNoZero(std::sin(theta2));
    const double theta1 = std::atan2(-sgn2 * r22, -sgn2 * r12);
    const double theta3 = std::atan2(sgn2 * r33, -sgn2 * r31);
    return {theta1, theta2, theta3};
  };

  return {solve(theta2_p), solve(theta2_m)};
}

std::array<double, 2> theta56TwoBranches(
    const Eigen::Matrix3d& Aw,
    const Eigen::Matrix3d& Bw,
    const Eigen::Matrix3d& Cw,
    const double phi,
    const double theta7) {
  const double sphi = std::sin(phi);
  const double cphi = std::cos(phi);
  const Eigen::Matrix3d R4_7 = Aw * sphi + Bw * cphi + Cw;

  const double r13 = R4_7(0, 2);
  const double r23 = R4_7(1, 2);
  const double r31 = R4_7(2, 0);
  const double r32 = R4_7(2, 1);
  const double r33 = R4_7(2, 2);

  constexpr double kTwoPi = 2.0 * M_PI;
  const double s6_abs = std::hypot(r31, r32);

  const auto candidate = [&](const double s6_sign) -> std::array<double, 3> {
    const double theta6 = std::atan2(s6_sign * s6_abs, r33);
    const double theta5 = std::atan2(s6_sign * r23, s6_sign * r13);
    const double theta7_hat = std::atan2(s6_sign * r32, s6_sign * (-r31));
    const double d = std::abs(std::remainder(theta7_hat - theta7, kTwoPi));
    return {theta5, theta6, d};
  };

  const auto c_pos = candidate(+1.0);
  const auto c_neg = candidate(-1.0);
  const auto best = (c_pos[2] <= c_neg[2]) ? c_pos : c_neg;

  return {best[0], best[1]};
}

}

std::vector<std::array<double, 7>> SolveAik7_ow_left(
    const Eigen::Isometry3d& T_base_ee,
    const double q7_rad,
    const Dm25Model& model,
    int elbow_sign) {
  std::vector<std::array<double, 7>> solutions;
  solutions.reserve(4);

  // Convert input pose into DH0 frame.
  const Eigen::Isometry3d T_0_ee = model.T_dh0_base * T_base_ee;
  const Eigen::Isometry3d T_0_7 = T_0_ee * model.T_ee_dh7;
  const Eigen::Matrix3d R_0_7 = T_0_7.linear();

  const double theta7 = q7_rad + model.dh[6].theta0_rad;
  const double c7 = std::cos(theta7);
  const double s7 = std::sin(theta7);

  // Compute T_0_6 from ee pose and given q7.
  const Eigen::Isometry3d T_6_7 =
      dhTStandardCS(c7, s7, model.dh[6], model.dh_alpha_cos[6], model.dh_alpha_sin[6]);
  const Eigen::Isometry3d T_0_6 = T_0_7 * T_6_7.inverse();

  // Wrist point Pw: joint6 axis point (frame5 origin), expressed in frame0.
  const Eigen::Vector3d p6_0 = T_0_6.translation();
  const Eigen::Vector3d x6_0 = T_0_6.linear().col(0);
  const Eigen::Vector3d Pw_0 = p6_0 - model.dh[5].a_m * x6_0; // assumes d6 == 0

  // Shoulder point Ps: joint2 position (for DM25 DH it is fixed at (0,0,d1) up to tiny a1).
  const Eigen::Vector3d Ps_0(0.0, 0.0, model.dh[0].d_m);

  const Eigen::Vector3d L_sw_0 = Pw_0 - Ps_0;
  const double L_sw_norm = L_sw_0.norm();

  const Eigen::Vector3d u_sw = L_sw_0 / L_sw_norm;
  const Eigen::Matrix3d u_sw_x = skew3(u_sw);

  // Elbow angle theta4: single solution controlled by elbow_sign.
  const auto theta4_opt = computeTheta4(L_sw_0, model, elbow_sign);
  if (!theta4_opt.has_value()) {
    return solutions; // 不可达，直接返回空
  }
  const double theta4 = theta4_opt.value();

  {
    // TJ_FX model uses theta3_ref = 0.
    const Eigen::Isometry3d T_2_3 =
        dhTStandard(0.0, model.dh[2], model.dh_alpha_cos[2], model.dh_alpha_sin[2]);
    const Eigen::Vector3d p_2_3 = T_2_3.translation();
    const Eigen::Matrix3d R_2_3 = T_2_3.linear();
    const Eigen::Vector3d L_23_3 = R_2_3.transpose() * p_2_3;

    const Eigen::Isometry3d T_3_4 =
        dhTStandard(theta4, model.dh[3], model.dh_alpha_cos[3], model.dh_alpha_sin[3]);
    const Eigen::Vector3d p_3_4 = T_3_4.translation();
    const Eigen::Matrix3d R_3_4 = T_3_4.linear();

    const Eigen::Vector3d L_45_4(0.0, 0.0, model.dh[4].d_m);
    const Eigen::Vector3d v3 = L_23_3 + p_3_4 + R_3_4 * L_45_4;

    const auto th12_0_branches = theta12_0TwoBranches(L_sw_0, v3);
    for (const auto& th12_0 : th12_0_branches) {
      const double theta1_0 = th12_0[0];
      const double theta2_0 = th12_0[1];

      Eigen::Isometry3d T_0_3_ref = Eigen::Isometry3d::Identity();
      T_0_3_ref = T_0_3_ref * dhTStandard(theta1_0, model.dh[0], model.dh_alpha_cos[0], model.dh_alpha_sin[0]);
      T_0_3_ref = T_0_3_ref * dhTStandard(theta2_0, model.dh[1], model.dh_alpha_cos[1], model.dh_alpha_sin[1]);
      T_0_3_ref =
          T_0_3_ref * dhTStandard(0.0, model.dh[2], model.dh_alpha_cos[2], model.dh_alpha_sin[2]);
      const Eigen::Matrix3d R_0_3_0 = T_0_3_ref.linear();

      const Eigen::Matrix3d As = u_sw_x * R_0_3_0;
      const Eigen::Matrix3d uuT = u_sw * u_sw.transpose();
      const Eigen::Matrix3d Bs = (Eigen::Matrix3d::Identity() - uuT) * R_0_3_0;
      const Eigen::Matrix3d Cs = uuT * R_0_3_0;

      const Eigen::Matrix3d Aw = R_3_4.transpose() * As.transpose() * R_0_7;
      const Eigen::Matrix3d Bw = R_3_4.transpose() * Bs.transpose() * R_0_7;
      const Eigen::Matrix3d Cw = R_3_4.transpose() * Cs.transpose() * R_0_7;

      const auto phi_branches = phiTwoBranches(Aw, Bw, Cw, s7, c7);
      for (const double phi : phi_branches) {
        const auto theta123_branches = theta123TwoBranches(As, Bs, Cs, phi);
        const auto th56 = theta56TwoBranches(Aw, Bw, Cw, phi, theta7);

        for (const auto& th123 : theta123_branches) {
          const std::array<double, 7> theta = {th123[0], th123[1], th123[2], theta4, th56[0], th56[1], theta7};

          std::array<double, 7> q{};
          for (size_t i = 0; i < 7; ++i) {
            q[i] = theta[i] - model.dh[i].theta0_rad;
          }
          q[6] = q7_rad; // enforce exact redundancy input
          solutions.push_back(q);
        }
      }
    }
  }

  // ----------------------------
  // Post-process: joint-limit filtering + de-duplication
  // ----------------------------
  //
  // Notes:
  // - IK 输出是 URDF q-space（rad），关节角在运动学上对 2*pi 周期等价；
  //   这里会尝试把 q[0..5] 折回到 URDF 限位区间内（若存在等价角度），再做限位检查。
  // - q7 轴是冗余输入：保持严格等于输入的 q7_rad，不做 2*pi 折回。
  const auto& q_limits = model.joint_limits.q_limits_rad;
  constexpr double kLimitTolRad = 1e-8;
  constexpr double kFkPosTolM = 1e-5;
  constexpr double kFkRotTolRad = 1e-6;

  size_t write_idx = 0;
  for (auto& q : solutions) {
    // Wrap q1..q6 (index 0..5) into limits where possible.
    for (size_t i = 0; i < 6; ++i) {
      q[i] = wrapAngleIntoRange(q[i], q_limits[i], kLimitTolRad);
    }

    // Keep redundancy axis exact.
    q[6] = q7_rad;

    bool ok = true;
    for (size_t i = 0; i < 7; ++i) {
      ok = ok && isWithinLimit(q[i], q_limits[i], kLimitTolRad);
    }
    if (!ok) {
      continue;
    }

    if (!passesForwardPoseValidation(q, T_base_ee, model, kFkPosTolM, kFkRotTolRad)) {
      continue;
    }

    solutions[write_idx++] = q;
  }

  solutions.resize(write_idx);
  return solutions;
}

} // namespace daimon_aik
