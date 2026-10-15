#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


def default_joint_names():
    return [
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
    ]


class InitialJointStateSeedNode(Node):
    def __init__(self):
        super().__init__("initial_joint_state_seed_node")

        self.declare_parameter("joint_topic", "/joint_states")
        self.declare_parameter("joint_names", default_joint_names())
        self.declare_parameter("publish_delay_sec", 0.5)
        self.declare_parameter("publish_period_sec", 0.1)
        self.declare_parameter("publish_count", 50)

        self.joint_topic = self.get_parameter("joint_topic").value
        self.joint_names = self.get_parameter("joint_names").value
        self.publish_delay_sec = max(0.01, float(self.get_parameter("publish_delay_sec").value))
        self.publish_period_sec = max(0.01, float(self.get_parameter("publish_period_sec").value))
        self.publish_count = max(1, int(self.get_parameter("publish_count").value))

        self.publisher = self.create_publisher(JointState, self.joint_topic, 10)
        self.publish_index = 0

        self.get_logger().info(
            f"Initial joint-state seed node started. "
            f"topic='{self.joint_topic}', joints={len(self.joint_names)}, "
            f"delay={self.publish_delay_sec:.2f}s, period={self.publish_period_sec:.2f}s, "
            f"count={self.publish_count}"
        )

        self.timer = self.create_timer(self.publish_delay_sec, self.publish_seed)

    def publish_seed(self):
        if self.publish_index >= self.publish_count:
            self.get_logger().info(
                f"Published {self.publish_count} zero joint-state seeds on {self.joint_topic}."
            )
            self.destroy_timer(self.timer)
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.joint_names)
        msg.position = [0.0] * len(self.joint_names)
        self.publisher.publish(msg)
        self.publish_index += 1

        if self.publish_index < self.publish_count:
            self.timer.reset()


def main():
    rclpy.init()
    node = InitialJointStateSeedNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
