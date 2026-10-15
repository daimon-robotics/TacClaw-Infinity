// Humble 使用 get_package_share_directory (返回 string), Jazzy+ 使用 get_package_share_path (返回 path)
#if __has_include(<ament_index_cpp/get_package_share_directory.hpp>)
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <filesystem>
namespace ament_index_cpp {
  inline std::filesystem::path get_package_share_path(const std::string& pkg) {
    return std::filesystem::path(get_package_share_directory(pkg));
  }
}
#else
#include <ament_index_cpp/get_package_share_path.hpp>
#endif
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <interactive_markers/interactive_marker_server.hpp>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/interactive_marker.hpp>
#include <visualization_msgs/msg/interactive_marker_control.hpp>
#include <visualization_msgs/msg/interactive_marker_feedback.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <wbc_msgs/msg/wbc_pose_expand.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kLeftArmPoseMask = 0x01u;
constexpr std::uint32_t kRightArmPoseMask = 0x02u;

enum class ArmSide {
  kLeft,
  kRight,
};

geometry_msgs::msg::Quaternion NormalizeQuaternion(
    const geometry_msgs::msg::Quaternion& quaternion) {
  const double norm = std::sqrt(
      quaternion.x * quaternion.x +
      quaternion.y * quaternion.y +
      quaternion.z * quaternion.z +
      quaternion.w * quaternion.w);
  geometry_msgs::msg::Quaternion normalized;
  if (norm <= 1e-12) {
    normalized.w = 1.0;
    return normalized;
  }
  normalized.x = quaternion.x / norm;
  normalized.y = quaternion.y / norm;
  normalized.z = quaternion.z / norm;
  normalized.w = quaternion.w / norm;
  return normalized;
}

std::string DefaultUrdfPath() {
  return (ament_index_cpp::get_package_share_path("daimon_tj_fx_description") /
          "urdf" / "daimon_tj_fx_dual_arm.urdf")
      .string();
}

}  // namespace

class WbcArmIkGoalMarkersNode : public rclcpp::Node {
 public:
  WbcArmIkGoalMarkersNode()
      : rclcpp::Node("wbc_arm_ik_goal_markers_node") {
    urdf_path_ = this->declare_parameter<std::string>("urdf_path", DefaultUrdfPath());
    frame_id_ = this->declare_parameter<std::string>("frame_id", "base_link");
    goal_topic_ = this->declare_parameter<std::string>("goal_topic", "/wbc_goal_pose");
    server_namespace_ =
        this->declare_parameter<std::string>("server_namespace", "wbc_arm_ik_goal_markers");
    publish_rate_hz_ = std::max(1.0, this->declare_parameter<double>("publish_rate_hz", 30.0));
    marker_scale_ = std::max(0.05, this->declare_parameter<double>("marker_scale", 0.22));

    goal_pub_ = create_publisher<wbc_msgs::msg::WbcPoseExpand>(
        goal_topic_, rclcpp::QoS(10).reliable());
    server_ = std::make_shared<interactive_markers::InteractiveMarkerServer>(
        server_namespace_,
        get_node_base_interface(),
        get_node_clock_interface(),
        get_node_logging_interface(),
        get_node_topics_interface(),
        get_node_services_interface());

    // This node only visualizes and publishes pose goals. Keep the startup path
    // independent from Pinocchio parsing so interactive markers stay reliable.
    initializeArmMarker(
        &left_arm_,
        ArmSide::kLeft,
        "goal_left",
        "Left EE Goal",
        declareInitialPoseParameter(
            "left_initial_position_xyz",
            {-0.413523, 0.477293, 0.286436},
            "left_initial_orientation_xyzw",
            {0.832131, 0.159111, 0.435860, -0.303756}));
    initializeArmMarker(
        &right_arm_,
        ArmSide::kRight,
        "goal_right",
        "Right EE Goal",
        declareInitialPoseParameter(
            "right_initial_position_xyz",
            {0.390907, -0.464485, -0.270779},
            "right_initial_orientation_xyzw",
            {-0.159092, 0.832132, 0.303749, 0.435870}));

    insertMarker(left_arm_);
    insertMarker(right_arm_);
    server_->applyChanges();
    publishGoalNow();

    const auto period = std::chrono::duration<double>(1.0 / publish_rate_hz_);
    publish_timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(period),
        std::bind(&WbcArmIkGoalMarkersNode::publishGoalNow, this));

    RCLCPP_INFO(
        get_logger(),
        "Started WBC arm IK goal markers. goal_topic='%s', server='/%s', frame='%s'",
        goal_topic_.c_str(),
        server_namespace_.c_str(),
        frame_id_.c_str());
  }

 private:
  struct ArmMarkerContext {
    ArmSide side{ArmSide::kLeft};
    std::string marker_name;
    std::string description;
    geometry_msgs::msg::Pose pose;
  };

  void initializeArmMarker(ArmMarkerContext* arm,
                           const ArmSide side,
                           const std::string& marker_name,
                           const std::string& description,
                           const geometry_msgs::msg::Pose& initial_pose) const {
    arm->side = side;
    arm->marker_name = marker_name;
    arm->description = description;
    arm->pose = initial_pose;
  }

  geometry_msgs::msg::Pose declareInitialPoseParameter(
      const std::string& position_param_name,
      const std::vector<double>& default_position,
      const std::string& orientation_param_name,
      const std::vector<double>& default_orientation) {
    const auto position =
        this->declare_parameter<std::vector<double>>(position_param_name, default_position);
    if (position.size() != 3u) {
      throw std::invalid_argument(position_param_name + " must contain exactly 3 values");
    }

    const auto orientation = this->declare_parameter<std::vector<double>>(
        orientation_param_name, default_orientation);
    if (orientation.size() != 4u) {
      throw std::invalid_argument(orientation_param_name + " must contain exactly 4 values");
    }

    geometry_msgs::msg::Pose pose;
    pose.position.x = position[0];
    pose.position.y = position[1];
    pose.position.z = position[2];
    pose.orientation.x = orientation[0];
    pose.orientation.y = orientation[1];
    pose.orientation.z = orientation[2];
    pose.orientation.w = orientation[3];
    pose.orientation = NormalizeQuaternion(pose.orientation);
    return pose;
  }

  void insertMarker(const ArmMarkerContext& arm) {
    visualization_msgs::msg::InteractiveMarker marker;
    marker.header.frame_id = frame_id_;
    marker.name = arm.marker_name;
    marker.description = arm.description;
    marker.scale = marker_scale_;

    {
      std::lock_guard<std::mutex> lock(marker_mutex_);
      marker.pose = arm.pose;
    }

    visualization_msgs::msg::Marker cube;
    cube.type = visualization_msgs::msg::Marker::CUBE;
    cube.scale.x = 0.05;
    cube.scale.y = 0.05;
    cube.scale.z = 0.05;
    if (arm.side == ArmSide::kLeft) {
      cube.color.r = 0.2f;
      cube.color.g = 0.6f;
      cube.color.b = 0.9f;
    } else {
      cube.color.r = 0.95f;
      cube.color.g = 0.55f;
      cube.color.b = 0.15f;
    }
    cube.color.a = 0.85f;

    visualization_msgs::msg::InteractiveMarkerControl visible_control;
    visible_control.always_visible = true;
    visible_control.interaction_mode =
        visualization_msgs::msg::InteractiveMarkerControl::NONE;
    visible_control.markers.push_back(cube);
    marker.controls.push_back(visible_control);

    auto add_axis = [&marker](const std::string& control_name,
                              const std::uint8_t interaction_mode,
                              const double ox,
                              const double oy,
                              const double oz,
                              const double ow) {
      visualization_msgs::msg::InteractiveMarkerControl control;
      control.orientation.x = ox;
      control.orientation.y = oy;
      control.orientation.z = oz;
      control.orientation.w = ow;
      control.orientation = NormalizeQuaternion(control.orientation);
      control.name = control_name;
      control.interaction_mode = interaction_mode;
      marker.controls.push_back(control);
    };

    add_axis("rotate_x", visualization_msgs::msg::InteractiveMarkerControl::ROTATE_AXIS, 1, 0, 0, 1);
    add_axis("move_x", visualization_msgs::msg::InteractiveMarkerControl::MOVE_AXIS, 1, 0, 0, 1);
    add_axis("rotate_y", visualization_msgs::msg::InteractiveMarkerControl::ROTATE_AXIS, 0, 1, 0, 1);
    add_axis("move_y", visualization_msgs::msg::InteractiveMarkerControl::MOVE_AXIS, 0, 1, 0, 1);
    add_axis("rotate_z", visualization_msgs::msg::InteractiveMarkerControl::ROTATE_AXIS, 0, 0, 1, 1);
    add_axis("move_z", visualization_msgs::msg::InteractiveMarkerControl::MOVE_AXIS, 0, 0, 1, 1);

    server_->insert(
        marker,
        [this, side = arm.side](
            const interactive_markers::InteractiveMarkerServer::FeedbackConstSharedPtr& feedback) {
          onMarkerFeedback(side, feedback);
        });
  }

  void onMarkerFeedback(
      const ArmSide side,
      const interactive_markers::InteractiveMarkerServer::FeedbackConstSharedPtr& feedback) {
    if (!feedback) {
      return;
    }
    if (feedback->event_type != visualization_msgs::msg::InteractiveMarkerFeedback::POSE_UPDATE &&
        feedback->event_type != visualization_msgs::msg::InteractiveMarkerFeedback::MOUSE_UP) {
      return;
    }

    std::lock_guard<std::mutex> lock(marker_mutex_);
    geometry_msgs::msg::Pose normalized_pose = feedback->pose;
    normalized_pose.orientation = NormalizeQuaternion(normalized_pose.orientation);
    if (side == ArmSide::kLeft) {
      left_arm_.pose = normalized_pose;
    } else {
      right_arm_.pose = normalized_pose;
    }

    publishGoalNowLocked();
  }

  void publishGoalNow() {
    std::lock_guard<std::mutex> lock(marker_mutex_);
    publishGoalNowLocked();
  }

  void publishGoalNowLocked() {
    wbc_msgs::msg::WbcPoseExpand goal_msg;
    const rclcpp::Time stamp = now();

    goal_msg.left_arm_goal_pose.header.stamp = stamp;
    goal_msg.left_arm_goal_pose.header.frame_id = frame_id_;
    goal_msg.left_arm_goal_pose.pose = left_arm_.pose;

    goal_msg.right_arm_goal_pose.header.stamp = stamp;
    goal_msg.right_arm_goal_pose.header.frame_id = frame_id_;
    goal_msg.right_arm_goal_pose.pose = right_arm_.pose;

    goal_msg.head_goal_pose.header.stamp = stamp;
    goal_msg.head_goal_pose.header.frame_id = frame_id_;
    goal_msg.head_goal_pose.pose.orientation.w = 1.0;

    goal_msg.hip_goal_pose.header.stamp = stamp;
    goal_msg.hip_goal_pose.header.frame_id = frame_id_;
    goal_msg.hip_goal_pose.pose.orientation.w = 1.0;

    goal_msg.pose_selection = kLeftArmPoseMask | kRightArmPoseMask;
    goal_pub_->publish(goal_msg);
  }

  rclcpp::Publisher<wbc_msgs::msg::WbcPoseExpand>::SharedPtr goal_pub_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
  std::shared_ptr<interactive_markers::InteractiveMarkerServer> server_;

  std::mutex marker_mutex_;
  std::string urdf_path_;
  std::string frame_id_;
  std::string goal_topic_;
  std::string server_namespace_;
  double publish_rate_hz_{30.0};
  double marker_scale_{0.22};
  ArmMarkerContext left_arm_;
  ArmMarkerContext right_arm_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<WbcArmIkGoalMarkersNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
