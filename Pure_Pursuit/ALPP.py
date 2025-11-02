#!/usr/bin/env python3
import numpy as np
import rclpy
import rclpy.logging
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from math import atan2, sqrt, pow, sin, exp
import pandas as pd
import time


class AdaptivePurePursuit(Node):

    def __init__(self):
        super().__init__('adaptive_pure_pursuit')

        # --- Subscribers and Publishers ---
        self.pos_sub = self.create_subscription(
            Odometry, '/odom', self.pos_callback, 10)
        self.thr_pub = self.create_publisher(
            Float32, '/autodrive/f1tenth_1/throttle_command', 10)
        self.str_pub = self.create_publisher(
            Float32, '/autodrive/f1tenth_1/steering_command', 10)
        self.goal_pub = self.create_publisher(Marker, '/goal', 10)
        self.cp_pub = self.create_publisher(Marker, '/cp', 10)
        self.race_pub = self.create_publisher(MarkerArray, '/raceline', 10)

        # --- Algorithm Parameters ---
        self.max_speed = 0.083
        self.min_speed = 0.06
        self.max_lookahead = 2.75
        self.min_lookahead = 1.75
        self.wheelbase = 0.33
        self.beta = 0.5  # Convex combination factor

        # --- State Variables ---
        self.current_quaternion = [0.0, 0.0, 0.0, 1.0]
        self.lookahead_distance = self.min_lookahead
        self.path = np.array([])
        self.previous_position = None
        self.previous_deviation = 0
        self.total_area = 0
        self.area_window = []
        self.window_size = 10
        self.position = None
        self.orientation = None
        
        # --- Default Control Values ---
        # (Used only until the first /odom message is received)
        self.control_velocity = 0.0015
        self.heading_angle = 0.01

        # --- Load Raceline ---
        self.load_raceline_csv(
            '/home/vatshvan/ros2_ws/src/alpp_pkg/Pure_Pursuit/raceline_n.csv')

        # --- One-Time Raceline Publisher ---
        # Publishes the raceline to RViz once after a 1s delay
        self.raceline_published = False
        self.raceline_timer = self.create_timer(
            1.0,  # 1-second delay
            self.publish_raceline_once
        )

    def load_raceline_csv(self, filename):
        try:
            self.path = pd.read_csv(filename)
            self.path = np.array([self.path]).reshape(-1, 2)
            self.path = self.path[::-1]
            for i in range(len(self.path)):
                self.path[i, 1] += 1.05
                self.path[i, 0] -= 3.6

            rotation_matrix = np.array([[-0.06, 1], [-1, 0]])
            self.path = np.dot(self.path, rotation_matrix.T)
            self.get_logger().info(f"Raceline loaded successfully from {filename}")
        except FileNotFoundError:
            self.get_logger().error(f"FATAL: Raceline file not found at {filename}")
            rclpy.shutdown()

    def publish_raceline_once(self):
        """
        Publishes the entire raceline as a MarkerArray one time.
        This timer cancels itself after successful publishing.
        """
        if self.raceline_published or len(self.path) == 0:
            self.raceline_timer.cancel()  # Stop the timer
            return

        marker_array = MarkerArray()
        for i in range(len(self.path)):
            marker = Marker()
            marker.header.frame_id = 'world'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.id = i
            marker.pose.position.x = self.path[i][0]
            marker.pose.position.y = self.path[i][1]
            marker.pose.position.z = 0.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)
        
        self.race_pub.publish(marker_array)
        self.raceline_published = True
        self.get_logger().info("Raceline published to RViz.")
        self.raceline_timer.cancel() # Stop the timer

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def update_lookahead(self, speed):
        normalized_speed = (speed - self.min_speed) / \
            (self.max_speed - self.min_speed)
        sigmoid_value = self.sigmoid(normalized_speed * 10 - 5)

        if speed < self.min_speed:
            self.lookahead_distance = self.min_lookahead
        else:
            scaled_lookahead = self.min_lookahead + sigmoid_value * \
                (self.max_lookahead - self.min_lookahead)
            self.lookahead_distance = min(self.max_lookahead, scaled_lookahead)

    def pos_callback(self, msg):
        """
        Main callback. Triggered every time a new /odom message is received.
        Calculates and publishes new control commands.
        """
        self.position = msg.pose.pose.position
        self.orientation = msg.pose.pose.orientation

        self.current_quaternion = [
            self.orientation.x, self.orientation.y, self.orientation.z, self.orientation.w]
        self.yaw = self.quaternion_to_yaw(self.current_quaternion)

        current_speed = msg.twist.twist.linear.x
        self.update_lookahead(current_speed)

        closest_point, goal_point = self.get_lookahead_point(self.position)
        
        if goal_point is not None:
            alpha = self.calculate_alpha(self.position, goal_point, self.yaw)
            self.heading_angle = self.calculate_heading_angle(alpha)
            area = self.calculate_deviation(self.position, closest_point)

            max_velocity_pp = self.calculate_max_velocity_pure_pursuit(
                self.calculate_curvature(alpha))
            min_deviation_pp = self.calculate_min_deviation_pure_pursuit(area)

            self.control_velocity = self.convex_combination(
                max_velocity_pp, min_deviation_pp, current_speed, area)
            
            # Now that values are updated, publish them
            self.publish_control_commands()
        
        # Removed "else" block: If no goal point is found,
        # just re-publish the previous command (which happens by default
        # as self.control_velocity isn't updated)

    def quaternion_to_yaw(self, quaternion):
        qx, qy, qz, qw = quaternion
        siny_cosp = 2*(qw * qz + qx * qy)
        cosy_cosp = 1 - 2*(qy * qy + qz * qz)
        yaw = atan2(siny_cosp, cosy_cosp)
        return yaw

    def get_lookahead_point(self, position):
        """
        Finds the closest point on the path and the lookahead goal point.
        """
        min_dist = float('inf')
        closest_point = None
        closest_point_index = -1

        # 1. Find the closest point and its index
        for i, point in enumerate(self.path):
            dist = sqrt(pow(point[0] - position.x, 2) +
                        pow(point[1] - position.y, 2))
            if dist < min_dist:
                min_dist = dist
                closest_point = point
                closest_point_index = i

        # 2. Find the goal point by searching forward from the closest point
        goal_point = None
        for i in range(closest_point_index, len(self.path)):
            point = self.path[i]
            dist_from_pos = sqrt(pow(point[0] - position.x, 2) +
                                 pow(point[1] - position.y, 2))
            
            # Find the first point that is *further* than the lookahead distance
            if dist_from_pos > self.lookahead_distance:
                goal_point = point
                break  # Found it

        # If no point is found (e.g., end of track), use the last point
        if goal_point is None:
            goal_point = self.path[-1]

        # --- Publish Markers for RViz (Goal and Closest Point) ---
        if closest_point is not None:
            marker = Marker()
            marker.header.frame_id = 'world'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = goal_point[0]
            marker.pose.position.y = goal_point[1]
            marker.pose.position.z = 0.0
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            self.goal_pub.publish(marker)

            marker2 = Marker()
            marker2.header.frame_id = 'world'
            marker2.header.stamp = self.get_clock().now().to_msg()
            marker2.type = Marker.SPHERE
            marker2.action = Marker.ADD
            marker2.pose.position.x = closest_point[0]
            marker2.pose.position.y = closest_point[1]
            marker2.pose.position.z = 0.0
            marker2.scale.x = 0.5
            marker2.scale.y = 0.5
            marker2.scale.z = 0.5
            marker2.color.a = 1.0
            marker2.color.r = 0.0
            marker2.color.g = 0.0
            marker2.color.b = 1.0
            self.cp_pub.publish(marker2)

        return closest_point, goal_point

    def calculate_alpha(self, position, goal_point, yaw):
        dy = goal_point[1] - position.y
        dx = goal_point[0] - position.x
        local_x = dx*np.cos(-yaw) - dy*np.sin(-yaw)
        local_y = dx*np.sin(-yaw) + dy*np.cos(-yaw)
        alpha = atan2(local_y, local_x)
        return alpha

    def calculate_heading_angle(self, alpha):
        heading_angle = atan2(2 * self.wheelbase *
                              sin(alpha), self.lookahead_distance)
        return heading_angle

    def calculate_curvature(self, alpha):
        curvature = 2 * sin(alpha) / self.lookahead_distance
        return curvature

    def calculate_deviation(self, position, closest_point):
        if closest_point is None:
            return self.total_area # Return last known area if no closest point
            
        deviation = sqrt(
            pow(closest_point[0] - position.x, 2) + pow(closest_point[1] - position.y, 2))

        if self.previous_position is not None:
            distance_traveled = sqrt(pow(position.x - self.previous_position.x, 2) +
                                     pow(position.y - self.previous_position.y, 2))
            area_increment = (
                deviation + self.previous_deviation) / 2 * distance_traveled

            self.area_window.append(area_increment)
            if len(self.area_window) > self.window_size:
                self.area_window.pop(0)

            self.total_area = sum(self.area_window)

        self.previous_position = position
        self.previous_deviation = deviation

        return self.total_area

    def calculate_max_velocity_pure_pursuit(self, curvature):
        max_velocity = sqrt(
            1 / abs(curvature)) if curvature != 0 else self.max_speed
        return min(self.max_speed, max_velocity)

    def calculate_min_deviation_pure_pursuit(self, area):
        if area > 0:
            min_deviation_velocity = self.max_speed / (1 + area)
        else:
            min_deviation_velocity = self.max_speed
        return min_deviation_velocity

    def convex_combination(self, max_velocity_pp, min_deviation_pp, current_speed, area):
        self.beta = self.adjust_beta(current_speed, area)

        control_velocity = self.beta * max_velocity_pp + \
            (1 - self.beta) * min_deviation_pp
        curvature = self.calculate_curvature(self.heading_angle)
        curv_diff = abs(curvature)
        control_velocity /= exp(2.4698 * (abs(curv_diff) ** 0.75))
        
        # Ensure velocity is within min/max bounds
        control_velocity = max(self.min_speed, min(self.max_speed, control_velocity))
        
        return control_velocity

    def adjust_beta(self, current_speed, area):
        if area < 1.0:
            return min(1.0, self.beta + 0.25)
        elif current_speed < self.max_speed * 0.4:
            return max(0.0, self.beta - 0.25)
        return self.beta

    def publish_control_commands(self):
        """
        Publishes the current throttle and steering commands.
        """
        throttle_msg = Float32()
        steering_msg = Float32()

        throttle_msg.data = float(self.control_velocity)
        steering_msg.data = float(self.heading_angle * 2.4)

        # Debug print
        print(f"Velocity: {self.control_velocity:.4f}, Heading: {self.heading_angle:.4f}")

        # Publish both commands
        self.thr_pub.publish(throttle_msg)
        self.str_pub.publish(steering_msg)


def main(args=None):
    rclpy.init(args=args)

    adaptive_pure_pursuit = AdaptivePurePursuit()

    try:
        rclpy.spin(adaptive_pure_pursuit)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        adaptive_pure_pursuit.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
