// C ABI boundary for the TJ_FX Aik7_ow analytic inverse kinematics core.
//
// Transform convention used by this bridge is row-major ^base T_ee: a point
// expressed in ee is mapped to base by p_base = R * p_ee + t.  The Python
// caller owns TCP->ee conversion and continuity / branch selection.

#include "Aik7_ow.h"
#include "current_urdf_model.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>
#include <cmath>
#include <exception>
#include <string>
#include <vector>

namespace {

thread_local std::string g_last_error;

void setError(const std::string& message) {
  g_last_error = message;
}

bool parseTransform(const double* values, Eigen::Isometry3d* transform) {
  if (values == nullptr || transform == nullptr) {
    setError("null transform pointer");
    return false;
  }
  Eigen::Matrix4d matrix;
  for (int row = 0; row < 4; ++row) {
    for (int column = 0; column < 4; ++column) {
      const double value = values[row * 4 + column];
      if (!std::isfinite(value)) {
        setError("transform contains a non-finite value");
        return false;
      }
      matrix(row, column) = value;
    }
  }
  if (!matrix.row(3).isApprox(Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0), 1e-9)) {
    setError("transform bottom row must be [0, 0, 0, 1]");
    return false;
  }
  const Eigen::Matrix3d rotation = matrix.block<3, 3>(0, 0);
  if (!rotation.transpose().isApprox(rotation.inverse(), 1e-7) ||
      std::abs(rotation.determinant() - 1.0) > 1e-7) {
    setError("transform rotation must be a proper SO(3) matrix");
    return false;
  }
  transform->matrix() = matrix;
  return true;
}

const daimon_aik::Dm25Model& modelForSide(const int side) {
  static const daimon_aik::Dm25Model left_model =
      tianji_aik7_model::CreateCurrentMuJoCoUrdfModel(0);
  static const daimon_aik::Dm25Model right_model =
      tianji_aik7_model::CreateCurrentMuJoCoUrdfModel(1);
  return side == 0 ? left_model : right_model;
}

Eigen::Isometry3d dhStandard(
    const double theta,
    const daimon_aik::Dm25Model::DhLink& link,
    const double cosine_alpha,
    const double sine_alpha) {
  const double cosine_theta = std::cos(theta);
  const double sine_theta = std::sin(theta);
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.linear() << cosine_theta, -sine_theta * cosine_alpha, sine_theta * sine_alpha,
      sine_theta, cosine_theta * cosine_alpha, -cosine_theta * sine_alpha,
      0.0, sine_alpha, cosine_alpha;
  transform.translation() = Eigen::Vector3d(
      link.a_m * cosine_theta, link.a_m * sine_theta, link.d_m);
  return transform;
}

Eigen::Isometry3d forwardBaseEe(
    const std::array<double, 7>& q,
    const daimon_aik::Dm25Model& model) {
  Eigen::Isometry3d transform = model.T_base_dh0;
  for (size_t index = 0; index < q.size(); ++index) {
    const double theta = q[index] + model.dh[index].theta0_rad;
    transform = transform * dhStandard(
        theta, model.dh[index], model.dh_alpha_cos[index], model.dh_alpha_sin[index]);
  }
  return transform * model.T_dh7_ee;
}

void writeTransform(const Eigen::Isometry3d& transform, double* output) {
  for (int row = 0; row < 4; ++row) {
    for (int column = 0; column < 4; ++column) {
      output[row * 4 + column] = transform.matrix()(row, column);
    }
  }
}

}  // namespace

extern "C" {

int tianji_aik7_api_version() {
  return 1;
}

const char* tianji_aik7_last_error() {
  return g_last_error.c_str();
}

// Forward kinematics is exported only to prove at startup that the vendored
// analytic model and the MuJoCo URDF share one fixed frame convention.  It is
// not used to drive the viewer.
int tianji_aik7_forward(
    const int side,
    const double* q,
    double* base_from_ee_row_major) {
  g_last_error.clear();
  if (side != 0 && side != 1) {
    setError("side must be 0 (left) or 1 (right)");
    return -1;
  }
  if (q == nullptr || base_from_ee_row_major == nullptr) {
    setError("q and output transform pointers must not be null");
    return -1;
  }
  std::array<double, 7> joint_position{};
  for (size_t index = 0; index < joint_position.size(); ++index) {
    if (!std::isfinite(q[index])) {
      setError("joint vector contains a non-finite value");
      return -1;
    }
    joint_position[index] = q[index];
  }
  try {
    writeTransform(forwardBaseEe(joint_position, modelForSide(side)), base_from_ee_row_major);
    return 0;
  } catch (const std::exception& exception) {
    setError(exception.what());
    return -3;
  } catch (...) {
    setError("unknown C++ exception");
    return -3;
  }
}

// Return the number of analytic branches written to q_out, zero when the
// target has no valid branch, or a negative error code.  side is 0=left,
// 1=right.  q_out has capacity rows of seven doubles, stored row-major.
int tianji_aik7_solve(
    const int side,
    const double* base_from_ee_row_major,
    const double q7_rad,
    const int elbow_sign,
    double* q_out,
    const int capacity) {
  g_last_error.clear();
  if (side != 0 && side != 1) {
    setError("side must be 0 (left) or 1 (right)");
    return -1;
  }
  if (!std::isfinite(q7_rad)) {
    setError("q7_rad must be finite");
    return -1;
  }
  if (q_out == nullptr || capacity <= 0) {
    setError("q_out must have positive capacity");
    return -1;
  }

  Eigen::Isometry3d base_from_ee = Eigen::Isometry3d::Identity();
  if (!parseTransform(base_from_ee_row_major, &base_from_ee)) {
    return -1;
  }

  try {
    const auto& model = modelForSide(side);
    const std::vector<std::array<double, 7>> solutions = side == 0
        ? daimon_aik::SolveAik7_ow_left(base_from_ee, q7_rad, model, elbow_sign)
        : daimon_aik::SolveAik7_ow_right(base_from_ee, q7_rad, model, elbow_sign);
    if (static_cast<int>(solutions.size()) > capacity) {
      setError("output capacity is smaller than the number of analytic branches");
      return -2;
    }
    for (size_t row = 0; row < solutions.size(); ++row) {
      for (size_t column = 0; column < solutions[row].size(); ++column) {
        q_out[row * 7 + column] = solutions[row][column];
      }
    }
    return static_cast<int>(solutions.size());
  } catch (const std::exception& exception) {
    setError(exception.what());
    return -3;
  } catch (...) {
    setError("unknown C++ exception");
    return -3;
  }
}

}  // extern "C"
