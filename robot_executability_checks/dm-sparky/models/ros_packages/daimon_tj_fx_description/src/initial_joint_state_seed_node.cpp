#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <algorithm>
#include <chrono>
#include <functional>
#include <string>
#include <vector>

namespace {

std::vector<std::string> DefaultJointNames() {
  return {
      "Joint1_L",
      "Joint2_L",
      "Joint3_L",
      "Joint4_L",
      "Joint5_L",
      "Joint6_L",
      "Joint7_L",
      "Joint1_R",
      "Joint2_R",
      "Joint3_R",
      "Joint4_R",
      "Joint5_R",
      "Joint6_R",
      "Joint7_R",
  };
}

}  // namespace

class InitialJointStateSeedNode : public rclcpp::Node {
 public:
  InitialJointStateSeedNode()
      : rclcpp::Node("initial_joint_state_seed_node") {
    joint_topic_ = declare_parameter<std::string>("joint_topic", "/joint_states");
    joint_names_ = declare_parameter<std::vector<std::string>>("joint_names", DefaultJointNames());
    publish_delay_sec_ = std::max(0.01, declare_parameter<double>("publish_delay_sec", 0.5));

    publisher_ = create_publisher<sensor_msgs::msg::JointState>(
        joint_topic_, rclcpp::QoS(10).reliable());

    const auto period = std::chrono::duration<double>(publish_delay_sec_);
    timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(period),
        std::bind(&InitialJointStateSeedNode::publishSeedJointState, this));

    RCLCPP_INFO(
        get_logger(),
        "Initial joint-state seed node started. topic='%s', joints=%zu, delay=%.2f s",
        joint_topic_.c_str(),
        joint_names_.size(),
        publish_delay_sec_);
  }

 private:
  void publishSeedJointState() {
    if (seed_sent_) {
      rclcpp::shutdown();
      return;
    }

    sensor_msgs::msg::JointState message;
    message.header.stamp = now();
    message.name = joint_names_;
    message.position.assign(joint_names_.size(), 0.0);
    publisher_->publish(message);
    seed_sent_ = true;
    RCLCPP_INFO(get_logger(), "Published one zero joint-state seed on %s.", joint_topic_.c_str());
  }

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::string joint_topic_;
  std::vector<std::string> joint_names_;
  double publish_delay_sec_{0.5};
  bool seed_sent_{false};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<InitialJointStateSeedNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
