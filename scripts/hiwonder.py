# hiwonder.py
"""
Hiwonder Robot Controller
-------------------------
Handles the control of the mobile base and 5-DOF robotic arm using commands received from the gamepad.
"""

import time
from math import sin, cos, atan2, radians, degrees, sqrt, acos
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

        self.home_position = [0, -85.04, -64.58, -69.54, 0, 0]  # degrees
        self.square_position = [0, -10, -64.58, -69.54, 0, 0]

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

        print(f"j_1: {cmd.arm_j1}")
        if cmd.arm_home:
            self.move_to_home_position()
        elif cmd.arm_j1:
            self.move_square()

        # print(f"---------------------------------------------------------------------")

        # self.set_base_velocity(cmd)
        self.set_arm_velocity(cmd)

        ######################################################################

        position = [0] * 3

        ######################################################################

        # print(
        #    f"[DEBUG] XYZ position: X: {round(position[0], 3)}, Y: {round(position[1], 3)}, Z: {round(position[2], 3)} \n"
        # )

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

    def calc_DH_matrices(self, theta=None):
        """Calculates all DH Matrices of the system"""
        # DH table parameters
        theta_i_table = [
            self.theta[0],
            self.theta[1],
            self.theta[2],
            self.theta[3],
            self.theta[4],
        ]
        if theta is not None:
            theta_i_table = [
                theta[0],
                theta[1],
                theta[2],
                theta[3],
                theta[4],
            ]
        d_table = [self.l1, 0, 0, 0, self.l5]
        r_table = [0, self.l2, self.l3, self.l4, 0]
        alpha_table = [np.pi / 2, np.pi, np.pi, 0, 0]
        for i in range(5):
            if i == 0:
                self.DH[i] = self.DH_matrix(
                    theta_i_table[i], d_table[i], r_table[i], alpha_table[i]
                ) @ self.DH_matrix(np.pi / 2, 0, 0, 0)
            elif i == 3:
                self.DH[i] = self.DH_matrix(
                    theta_i_table[i], d_table[i], r_table[i], alpha_table[i]
                ) @ self.DH_matrix(-np.pi / 2, 0, 0, -np.pi / 2)
            else:
                self.DH[i] = self.DH_matrix(
                    theta_i_table[i], d_table[i], r_table[i], alpha_table[i]
                )

    def calc_numerical_ik(self, EE, tol=0.01, ilimit=50):
        """Calculate numerical inverse kinematics based on input coordinates."""

        ########################################

        # define initial guess
        """original_points = [
            self.ee.x,
            self.ee.y,
            self.ee.z,
            self.ee.rotx,
            self.ee.roty,
            self.ee.rotz,
        ]
        epsilon = 0.001
        desired_coords = [EE.x, EE.y, EE.z, EE.rotx, EE.roty, EE.rotz]
        points = original_points

        # calc initial variable
        error_var = [
            des_i - theta_i for des_i, theta_i in zip(desired_coords, points)
        ]  # 1x6 matrix
        i = 0
        while (sum(np.absolute(error_var)) / 6 >= epsilon) & (i < 3000):
            pseudojacobian = self.calc_pseduojacobian()  # 6x5 matrix
            new_points = pseudojacobian @ np.transpose(np.array(error_var))
            self.theta = self.theta + new_points  # 1x5 matrix \
            self.calc_points(self.theta)
            points = [
                self.ee.x,
                self.ee.y,
                self.ee.z,
                self.ee.rotx,
                self.ee.roty,
                self.ee.rotz,
            ]
            error_var = [
                des_i - theta_i for des_i, theta_i in zip(desired_coords, points)
            ]
            i += 1

        [EE.x, EE.y, EE.z, EE.rotx, EE.roty, EE.rotz] = original_points
        ########################################

        # Recompute robot points based on updated joint angles
        self.calc_forward_kinematics(self.theta, radians=True)"""

    def calc_analytical_inverse_kinematics(self, x, y, z, rot):
        """
        Calculates angle of each joint given x, y, z, and rotation.
        """
        theta = [0, -85.04, -64.58, -69.54, 0, 0]

        theta[0] = atan2(y, x)
        rot_z_theta1 = np.array(
            [
                [cos(theta[0]), -sin(theta[0]), 0],
                [sin(theta[0]), cos(theta[0]), 0],
                [0, 0, 1],
            ]
        )
        rotz = radians(rot)
        rot_y = np.array(
            [
                [cos(np.pi / 2 + rotz), 0, sin(np.pi / 2 + rotz)],
                [0, 1, 0],
                [-sin(np.pi / 2 + rotz), 0, cos(np.pi / 2 + rotz)],
            ]
        )
        k = np.transpose(np.array([[0, 0, 1]]))
        r_06 = rot_z_theta1 @ rot_y
        t_35 = (self.l4 + self.l5) * r_06 @ k

        p_wrist_x = x - t_35[0]
        p_wrist_y = y - t_35[1]
        p_wrist_z = z - t_35[2]

        rx = sqrt(p_wrist_x**2 + p_wrist_y**2)
        ry = p_wrist_z - self.l1

        theta[2] = acos(
            (rx**2 + ry**2 - self.l2**2 - self.l3**2) / (2 * self.l2 * self.l3)
        )
        alpha = atan2(self.l2 * sin(theta[2]), self.l2 + self.l3 * cos(theta[2]))
        gamma = atan2(ry, rx)
        theta[1] = (gamma - alpha) - (np.pi / 2)
        theta[2] = -theta[2]

        self.calc_DH_matrices(theta)
        r_03 = (self.DH[0] @ self.DH[1] @ self.DH[2])[:3, :3]
        r_35 = np.transpose(r_03) @ r_06

        theta[3] = atan2(r_35[0][0], r_35[0][2])

        theta = [degrees(i) for i in theta]

        print(theta)

        return theta

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

        # get thetalist_dot
        thetalist_dot = new_jacobian @ vel
        # normalize thetalist_dot and scale it by a factor
        thetalist_dot = thetalist_dot / (np.max(np.absolute(thetalist_dot)) + 0.01) / 6
        # turn thetalist_dot into degrees
        thetalist_dot = [degrees(theta) for theta in thetalist_dot]

        ######################################################################

        # print(f"[DEBUG] Current thetalist (deg) = {self.joint_values}")
        # print(
        #    f"[DEBUG] linear vel: {[round(vel[0], 3), round(vel[1], 3), round(vel[2], 3)]}"
        # )
        # print(f"[DEBUG] thetadot (deg/s) = {[round(td,2) for td in thetalist_dot]}")

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
        # print(f"[DEBUG] Commanded thetalist (deg) = {new_thetalist}")

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

        # print(f"[DEBUG] Moving joint {joint_id} to {theta}° ({pulse} pulse)")
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
        self.set_joint_values(
            self.calc_analytical_inverse_kinematics(0.29, 0, 0.255, 14),
            duration=500,
        )
        time.sleep(2.0)
        print(f"Arrived at home position: {self.joint_values} \n")
        time.sleep(1.0)
        print(f"------------------- System is now ready!------------------- \n")

    def move_square(self):
        """
        Move in square motion
        """
        print(f"Moving to square position...")
        self.set_joint_values(
            self.calc_analytical_inverse_kinematics(0.29, -0.0762, 0.331, -30),
            duration=500,
        )
        time.sleep(5.0)
        self.set_joint_values(
            self.calc_analytical_inverse_kinematics(0.29, 0.0762, 0.331, -30),
            duration=500,
        )
        time.sleep(5.0)
        self.set_joint_values(
            self.calc_analytical_inverse_kinematics(0.29, 0.0762, 0.179, 30),
            duration=500,
        )
        time.sleep(5.0)
        self.set_joint_values(
            self.calc_analytical_inverse_kinematics(0.29, -0.0762, 0.179, 30),
            duration=500,
        )
        time.sleep(5.0)
        print(f"-------------------Moved to Square!------------------- \n")
        time.sleep(10.0)

    # -------------------------------------------------------------
    # Utility Functions
    # -------------------------------------------------------------

    def calc_pseduojacobian(self):
        self.calc_DH_matrices()
        self.T_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            self.T_cumulative.append(self.T_cumulative[-1] @ self.DH[i])
        J_l = []
        J_w = []

        for i in range(4):
            k = np.array([0, 0, 1])
            rot_matrix = self.T_cumulative[i][:3, :3]
            z = rot_matrix @ k
            r = (self.points[5] - self.points[i])[:3]
            J_w.append(z)
            J_l.append(np.cross(z, r))

        # Construct the Jacobian matrix
        J_l.append([0, 0, 0])  # Adding a zero row for completeness
        J_w.append([0, 0, 0])  # Adding a zero row for completeness

        jacobian_l = np.transpose(np.array(J_l))
        jacobian_w = np.transpose(np.array(J_w))
        jacobian = np.concatenate((jacobian_l, jacobian_w), axis=0)

        # Calculate pseudo-Jacobian matrix using the transpose method
        if np.linalg.matrix_rank(jacobian) < min(jacobian.shape):
            # Handle near singularity or underdetermined system

            # Use a regularized pseudo-inverse calculation
            new_jacobian = np.transpose(jacobian) @ np.linalg.inv(
                jacobian @ np.transpose(jacobian) + np.eye(jacobian.shape[0]) * 1e-5
            )
        else:
            # Calculate pseudo-Jacobian matrix using the transpose method
            new_jacobian = np.transpose(jacobian) @ np.linalg.inv(
                jacobian @ np.transpose(jacobian)
            )

        return new_jacobian

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
