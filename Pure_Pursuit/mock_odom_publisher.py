#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Float32  # <-- Import steering message
import numpy as np
import math

def quaternion_from_euler(roll, pitch, yaw):
    """
    Converts euler roll, pitch, yaw to a Quaternion.
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q

class MockOdomPublisher(Node):

    def __init__(self):
        super().__init__('mock_odom_publisher')
        self.publisher_ = self.create_publisher(Odometry, '/odom', 10)
        
        # --- This is new ---
        # Create a subscriber to listen to the steering commands from alpp_node
        self.steering_sub_ = self.create_subscription(
            Float32,
            '/autodrive/f1tenth_1/steering_command',
            self.steering_callback,
            10)
        # --- End new ---

        self.timer_period = 0.05  # 20 Hz
        self.timer = self.create_timer(self.timer_period, self.publish_odom)
        
        # --- Updated State Variables ---
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0  # Robot's current angle (yaw)
        self.speed = 4.0  # Robot's speed (m/s)
        self.steering_angle = 0.0  # Current steering angle
        self.wheelbase = 0.33  # Wheelbase of the "car" (must match alpp_node)
        # --- End updated ---
        
        self.get_logger().info('Mock Odometry Publisher (CLOSED LOOP) has started.')

    def steering_callback(self, msg):
        # --- This is new ---
        # When we receive a steering command, store it
        self.steering_angle = msg.data
        # --- End new ---

    def publish_odom(self):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'

        # --- This is new: Simple car physics ---
        # Calculate the change in angle (yaw)
        delta_theta = (self.speed * math.tan(self.steering_angle)) / self.wheelbase * self.timer_period
        
        # Update the robot's state
        self.theta += delta_theta
        self.x += self.speed * math.cos(self.theta) * self.timer_period
        self.y += self.speed * math.sin(self.theta) * self.timer_period
        # --- End new physics ---

        # Set the position
        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.y = self.y
        msg.pose.pose.position.z = 0.0
        
        # Set the orientation from our yaw angle
        msg.pose.pose.orientation = quaternion_from_euler(0.0, 0.0, self.theta)

        # Report the current linear and angular velocity
        msg.twist.twist.linear.x = self.speed
        msg.twist.twist.linear.y = 0.0
        msg.twist.twist.angular.z = delta_theta / self.timer_period # (this is yaw rate)

        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    mock_odom_publisher = MockOdomPublisher()
    rclpy.spin(mock_odom_publisher)
    
    mock_odom_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
