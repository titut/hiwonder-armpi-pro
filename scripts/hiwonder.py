# hiwonder.py
"""
Hiwonder Robot Controller
-------------------------
Handles the control of the mobile base and 5-DOF robotic arm using commands received from the gamepad.
"""

import time
from math import sin, cos, radians
import numpy as np
from board_controller import BoardController
from servo_bus_controller import ServoBusController
import utils as ut

# Robot base constants
WHEEL_RADIUS = 0.047  # meters
BASE_LENGTH_X = 0.096  # meters
BASE_LENGTH_Y = 0.105  # meters


class HiwonderRobot:
    def __init__(self):
        """Initialize motor controllers, servo bus, and default robot states."""
        self.board = BoardController()
        self.servo_bus = ServoBusController()

        # arm angles
        self.joint_values = [0, 0, 90, 90, 0, 0]  # degrees
        self.theta = [radians(i) for i in self.joint_values]
        self.DH = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

        # arm lengths
        self.l1, self.l2, self.l3, self.l4, self.l5 = 0.155, 0.099, 0.095, 0.055, 0.105

        self.home_position = [0, -90, -90, -90, 0, 0]  # degrees
        self.joint_limits = [
            [-120, 120],
            [-90, 90],
            [-120, 120],
            [-100, 100],
            [-90, 90],
            [-120, 30],
        ]
        self.joint_control_delay = 0.2  # secs
        self.speed_control_delay = 0.2

        self.move_to_home_position()

    # -------------------------------------------------------------
    # Methods for interfacing with the mobile base
    # -------------------------------------------------------------

    def set_robot_commands(self, cmd: ut.GamepadCmds):
        """Updates robot base and arm based on gamepad commands.

        Args:
            cmd (GamepadCmds): Command data class with velocities and joint commands.
        """

        if cmd.arm_home:
            self.move_to_home_position()

        print(f"---------------------------------------------------------------------")

        # self.set_base_velocity(cmd)
        self.set_arm_velocity(cmd)

        ######################################################################

        position = [0] * 3

        ######################################################################

        print(
            f"[DEBUG] XYZ position: X: {round(position[0], 3)}, Y: {round(position[1], 3)}, Z: {round(position[2], 3)} \n"
        )

    def set_base_velocity(self, cmd: ut.GamepadCmds):
        """Computes wheel speeds based on joystick input and sends them to the board"""
        """
        motor3 w0|  ↑  |w1 motor1
                 |     |
        motor4 w2|     |w3 motor2
        
        """
        ######################################################################
        # insert your code for finding "speed"

        speed = [0] * 4

        ######################################################################

        # Send speeds to motors
        self.board.set_motor_speed(speed)
        time.sleep(self.speed_control_delay)

    # -------------------------------------------------------------
    # Methods for interfacing with the 5-DOF robotic arm
    # -------------------------------------------------------------

    def DH_matrix(self, theta, d, r, alpha):
        """Calculates DH matrix based on given arguments"""
        return np.array(
            [
                [
                    cos(theta),
                    -sin(theta) * cos(alpha),
                    sin(theta) * sin(alpha),
                    r * cos(theta),
                ],
                [
                    sin(theta),
                    cos(theta) * cos(alpha),
                    -cos(theta) * sin(alpha),
                    r * sin(theta),
                ],
                [0, sin(alpha), cos(alpha), d],
                [0, 0, 0, 1],
            ]
        )

    def calc_DH_matrices(self):
        """Calculates all DH Matrices of the system"""
        # DH table parameters
        theta_i_table = [
            self.theta[0],
            self.theta[1] - radians(90),
            self.theta[2] * -1,
            self.theta[3],
            -np.pi / 2,
        ]
        d_table = [self.l1, 0, 0, 0, 0]
        r_table = [0, self.l2, self.l3, self.l4, 0]
        alpha_table = [np.pi / 2, 0, 0, 0, -np.pi / 2]
        for i in range(5):
            self.DH[i] = self.DH_matrix(
                theta_i_table[i], d_table[i], r_table[i], alpha_table[i]
            )

    def set_arm_velocity(self, cmd: ut.GamepadCmds):
        """Calculates and sets new joint angles from linear velocities.

        Args:
            cmd (GamepadCmds): Contains linear velocities for the arm.
        """
        vel = [cmd.arm_vx, cmd.arm_vy, cmd.arm_vz]

        ######################################################################
        # insert your code for finding "thetalist_dot"

        self.theta = [radians(i) for i in self.joint_values]

        self.calc_DH_matrices()
        T_cumulative = [np.eye(4)]
        for i in range(5):
            if i == 4:
                T_cumulative.append(
                    T_cumulative[-1]
                    @ self.DH[i]
                    @ self.DH_matrix(self.theta[4], self.l5, 0, 0)
                )
            else:
                T_cumulative.append(T_cumulative[-1] @ self.DH[i])
        J = []

        points = [
            np.array([0, 0, 0, 1]),
            np.array([0, 0, 0, 1]),
            np.array([0, 0, 0, 1]),
            np.array([0, 0, 0, 1]),
            np.array([0, 0, 0, 1]),
            np.array([0, 0, 0, 1]),
        ]
        for i in range(1, 6):
            points[i] = T_cumulative[i] @ points[0]

        for i in range(4):
            k = np.array([0, 0, 1])
            rot_matrix = T_cumulative[i][:3, :3]
            z = rot_matrix @ k
            r = (points[5] - points[i])[:3]
            J.append(np.cross(z, r))

        jacobian = np.transpose(np.array([J[0], J[1], J[2], J[3], [0, 0, 0]]))
        new_jacobian = np.transpose(jacobian) @ np.linalg.inv(
            jacobian @ np.transpose(jacobian)
        )

        thetalist_dot = new_jacobian @ vel
        thetalist_dot = thetalist_dot / 5 / (np.max(thetalist_dot) + 1)

        ######################################################################

        print(f"[DEBUG] Current thetalist (deg) = {self.joint_values}")
        print(
            f"[DEBUG] linear vel: {[round(vel[0], 3), round(vel[1], 3), round(vel[2], 3)]}"
        )
        print(f"[DEBUG] thetadot (deg/s) = {[round(td,2) for td in thetalist_dot]}")

        # Update joint angles
        dt = 0.5  # Fixed time step
        K = 1600  # mapping gain for individual joint control
        new_thetalist = [0.0] * 6

        # linear velocity control
        for i in range(5):
            new_thetalist[i] = self.joint_values[i] + dt * thetalist_dot[i]
        # individual joint control
        new_thetalist[0] += dt * K * cmd.arm_j1
        new_thetalist[1] += dt * K * cmd.arm_j2
        new_thetalist[2] += dt * K * cmd.arm_j3
        new_thetalist[3] += dt * K * cmd.arm_j4
        new_thetalist[4] += dt * K * cmd.arm_j5
        new_thetalist[5] = self.joint_values[5] + dt * K * cmd.arm_ee

        new_thetalist = [round(theta, 2) for theta in new_thetalist]
        print(f"[DEBUG] Commanded thetalist (deg) = {new_thetalist}")

        # set new joint angles
        self.set_joint_values(new_thetalist, radians=False)

    def set_joint_value(self, joint_id: int, theta: float, duration=250, radians=False):
        """Moves a single joint to a specified angle"""
        if not (1 <= joint_id <= 6):
            raise ValueError("Joint ID must be between 1 and 6.")

        if radians:
            theta = np.rad2deg(theta)

        theta = self.enforce_joint_limits(theta, joint_id=joint_id)
        self.joint_values[joint_id] = theta

        pulse = self.angle_to_pulse(theta)
        self.servo_bus.move_servo(joint_id, pulse, duration)

        print(f"[DEBUG] Moving joint {joint_id} to {theta}° ({pulse} pulse)")
        time.sleep(self.joint_control_delay)

    def set_joint_values(self, thetalist: list, duration=250, radians=False):
        """Moves all arm joints to the given angles.

        Args:
            thetalist (list): Target joint angles in degrees.
            duration (int): Movement duration in milliseconds.
        """
        if len(thetalist) != 6:
            raise ValueError("Provide 6 joint angles.")

        if radians:
            thetalist = [np.rad2deg(theta) for theta in thetalist]

        thetalist = self.enforce_joint_limits(thetalist)
        self.joint_values = thetalist  # updates joint_values with commanded thetalist
        thetalist = self.remap_joints(
            thetalist
        )  # remap the joint values from software to hardware

        for joint_id, theta in enumerate(thetalist, start=1):
            pulse = self.angle_to_pulse(theta)
            self.servo_bus.move_servo(joint_id, pulse, duration)

    def enforce_joint_limits(self, thetalist: list) -> list:
        """Clamps joint angles within their hardware limits.

        Args:
            thetalist (list): List of target angles.

        Returns:
            list: Joint angles within allowable ranges.
        """
        return [
            np.clip(theta, *limit) for theta, limit in zip(thetalist, self.joint_limits)
        ]

    def move_to_home_position(self):
        print(f"Moving to home position...")
        self.set_joint_values(self.home_position, duration=800)
        time.sleep(2.0)
        print(f"Arrived at home position: {self.joint_values} \n")
        time.sleep(1.0)
        print(f"------------------- System is now ready!------------------- \n")

    # -------------------------------------------------------------
    # Utility Functions
    # -------------------------------------------------------------

    def angle_to_pulse(self, x: float):
        """Converts degrees to servo pulse value"""
        hw_min, hw_max = 0, 1000  # Hardware-defined range
        joint_min, joint_max = -150, 150
        return int(
            (x - joint_min) * (hw_max - hw_min) / (joint_max - joint_min) + hw_min
        )

    def pulse_to_angle(self, x: float):
        """Converts servo pulse value to degrees"""
        hw_min, hw_max = 0, 1000  # Hardware-defined range
        joint_min, joint_max = -150, 150
        return round(
            (x - hw_min) * (joint_max - joint_min) / (hw_max - hw_min) + joint_min, 2
        )

    def stop_motors(self):
        """Stops all motors safely"""
        self.board.set_motor_speed([0] * 4)
        print("[INFO] Motors stopped.")

    def remap_joints(self, thetalist: list):
        """Reorders angles to match hardware configuration.

        Args:
            thetalist (list): Software joint order.

        Returns:
            list: Hardware-mapped joint angles.

        Note: Joint mapping for hardware
            HARDWARE - SOFTWARE
            joint[0] = gripper/EE
            joint[1] = joint[5]
            joint[2] = joint[4]
            joint[3] = joint[3]
            joint[4] = joint[2]
            joint[5] = joint[1]
        """
        return thetalist[::-1]
