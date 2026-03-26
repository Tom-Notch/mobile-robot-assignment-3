#!/usr/bin/env python3

import copy

import numpy as np

from quadrotor_simulator_py.quadrotor_control.state import State
from quadrotor_simulator_py.utils.pose import Pose


def _velocity_constraint_matrix(T: float) -> np.ndarray:
    """8x8 matrix for v(t)=sum_{i=0}^7 c_i t^i with b = [v0,v0',v0'',v0''', vT,...]."""
    A: np.ndarray = np.zeros((8, 8))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 2.0
    A[3, 3] = 6.0
    for i in range(8):
        A[4, i] = T**i
    for i in range(1, 8):
        A[5, i] = i * T ** (i - 1)
    for i in range(2, 8):
        A[6, i] = i * (i - 1) * T ** (i - 2)
    for i in range(3, 8):
        A[7, i] = i * (i - 1) * (i - 2) * T ** (i - 3)
    return A


def _poly_derivative_ascending(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=float).flatten()
    if len(c) <= 1:
        return np.array([0.0])
    return np.array([(i + 1) * c[i + 1] for i in range(len(c) - 1)])


def _poly_eval_ascending(c: np.ndarray, t: float) -> float:
    c = np.asarray(c, dtype=float).flatten()
    return float(np.polyval(c[::-1], t))


def _analytic_state_at_time(
    s0: State,
    vx: float,
    omega: float,
    vz: float,
    t: float,
) -> State:
    """World-frame state at time t for constant body command [vx,0,vz] and yaw rate omega."""
    out: State = State()
    yaw0: float = float(s0.yaw)
    vb: np.ndarray = np.array([[vx], [0.0], [vz]])

    def v_w(tt: float) -> np.ndarray:
        th: float = yaw0 + omega * tt
        c: float = np.cos(th)
        s: float = np.sin(th)
        R: np.ndarray = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return R @ vb

    def a_w(tt: float) -> np.ndarray:
        th: float = yaw0 + omega * tt
        c: float = np.cos(th)
        s: float = np.sin(th)
        return np.array(
            [
                [-omega * s * vx],
                [omega * c * vx],
                [0.0],
            ]
        )

    def j_w(tt: float) -> np.ndarray:
        th: float = yaw0 + omega * tt
        c: float = np.cos(th)
        s: float = np.sin(th)
        return np.array(
            [
                [-(omega**2) * c * vx],
                [-(omega**2) * s * vx],
                [0.0],
            ]
        )

    def sn_w(tt: float) -> np.ndarray:
        th: float = yaw0 + omega * tt
        c: float = np.cos(th)
        s: float = np.sin(th)
        return np.array(
            [
                [(omega**3) * s * vx],
                [-(omega**3) * c * vx],
                [0.0],
            ]
        )

    out.vel = v_w(t)
    out.acc = a_w(t)
    out.jerk = j_w(t)
    out.snap = sn_w(t)

    dx: float
    dy: float
    if abs(omega) < 1e-12:
        dx = float(np.cos(yaw0) * vx * t)
        dy = float(np.sin(yaw0) * vx * t)
    else:
        th0: float = yaw0
        th1: float = yaw0 + omega * t
        dx = float(vx / omega * (np.sin(th1) - np.sin(th0)))
        dy = float(-vx / omega * (np.cos(th1) - np.cos(th0)))
    dz: float = float(vz * t)

    out.pos = s0.pos.copy() + np.array([[dx], [dy], [dz]])
    out.yaw = yaw0 + omega * t
    out.dyaw = omega
    out.d2yaw = 0.0
    out.d3yaw = 0.0
    out.rot = np.eye(3)
    out.angvel = np.zeros((3, 1))
    out.angacc = np.zeros((3, 1))
    return out


def _analytic_end_state(
    s0: State, vx: float, omega: float, vz: float, T: float
) -> State:
    """World-frame end state for constant body-frame command [vx,0,vz] and yaw rate omega."""
    return _analytic_state_at_time(s0, vx, omega, vz, T)


class ForwardArcTrajectory(object):

    def __init__(self, curr_ref: State, cmd: np.ndarray, T: float) -> None:
        """Calculates the 8th order forward arc motion primitive coefficients

        Args:
            curr_ref is a state object for where to sample the trajectory
            cmd is a 3x1 numpy array consisting of [vel_x, omega, vel_z]
            T is the primitive duration

        Output:
            coeffs: 4x9 world frame coefficients
        """

        self.T: float = float(T)
        vx: float = float(cmd[0, 0])
        omega: float = float(cmd[1, 0])
        vz: float = float(cmd[2, 0])

        s0: State = copy.deepcopy(curr_ref)
        e_end: State = _analytic_end_state(s0, vx, omega, vz, T)
        e_end.snap = np.zeros((3, 1))

        vel_c: np.ndarray = self.calculate_coefficients_from_contraints(s0, e_end)
        pos_c: np.ndarray = self.integrate_coefficients(vel_c)
        pos_c[0, 0] += float(s0.pos[0, 0])
        pos_c[1, 0] += float(s0.pos[1, 0])
        pos_c[2, 0] += float(s0.pos[2, 0])
        pos_c[3, 0] += float(s0.yaw)
        self.coeffs: np.ndarray = pos_c
        self.initialized: bool = True

    def coeffs_bodyf2worldf(self, coeffs: np.ndarray, Twb0: Pose) -> np.ndarray:
        """This is an optional function you can call within the __init__ funtion
        to convert coefficients from body frame to world frame.
        Args:
            coeffs: 4x9 numpy array of coefficients in body frame
            Twb0: Pose() object representing the initial pose of the quadrotor
        Output:
            coeffs_world_frame: 4x9 numpy array of coefficients in world frame
        """
        R: np.ndarray = Twb0.get_so3()
        t: np.ndarray = Twb0.translation().reshape(3, 1)
        out: np.ndarray = np.zeros((4, 9))
        for k in range(9):
            col: np.ndarray = coeffs[0:3, k].reshape(3, 1)
            out[0:3, k : k + 1] = R @ col
        out[3, :] = coeffs[3, :]
        out[0, 0] += t[0, 0]
        out[1, 0] += t[1, 0]
        out[2, 0] += t[2, 0]
        return out

    def integrate_coefficients(self, cin: np.ndarray) -> np.ndarray:
        """This function is an optional helper function you can call within
            the __init__ function. It integrates the coefficients passed.
            Coefficients are in the form of c0 + c1*t + c2*t*t + ... + c7*t^7.

        Args:
            cin: 4x8 matrix of coefficients for x, y, z, yaw

        Output:
            cout: 4x9 matrix of coefficients where the entry in
                  s[0,:] consists of a zero.
        """
        cout: np.ndarray = np.zeros((4, 9))
        for row in range(4):
            for j in range(8):
                cout[row, j + 1] = cin[row, j] / float(j + 1)
        return cout

    def calculate_coefficients_from_contraints(self, s: State, f: State) -> np.ndarray:
        """This function is an optional helper function you can call within
            the __init__ function. Given a initial constraints and final constraints,
            this function calculates the coefficients for the forward arc motion
            primitives. This function calls the optional get_coefficients function.
            Coefficients are in the form of c0 + c1*t + c2*t*t + ... + c7*t^7.
            Constraints are passed in in the body frame.

        Args:
            s: initial constraints State() class instance
            f: final constraints State() class instance

        Output:
            coeffs: 4x8 matrix of coefficients in the body frame
        """

        coeffs: np.ndarray = np.zeros((4, 8))
        for axis in range(3):
            b: np.ndarray = np.array(
                [
                    s.vel[axis, 0],
                    s.acc[axis, 0],
                    s.jerk[axis, 0],
                    s.snap[axis, 0],
                    f.vel[axis, 0],
                    f.acc[axis, 0],
                    f.jerk[axis, 0],
                    f.snap[axis, 0],
                ]
            )
            coeffs[axis, :] = self.get_coefficients(b.reshape(8, 1)).flatten()
        b = np.array(
            [
                s.dyaw,
                s.d2yaw,
                s.d3yaw,
                0.0,
                f.dyaw,
                f.d2yaw,
                f.d3yaw,
                0.0,
            ]
        )
        coeffs[3, :] = self.get_coefficients(b.reshape(8, 1)).flatten()
        return coeffs

    def get_coefficients(self, b: np.ndarray) -> np.ndarray:
        """This function is an optional helper function you can call within
            the calculate_coefficients_from_contraints function.
            Given the endpoints constraints in vector form (b), this function
            will calculate the solution to Ax=b we discussed in the Quadrotor Planning
            II lecture for forward arc motion primitives.

        Args:
            b: 8x1 np array of endpoint constraints

        Output:
            x: 8x1 nparray of coefficients
        """
        A: np.ndarray = _velocity_constraint_matrix(self.T)
        return np.linalg.solve(A, b)

    def get_ref(self, t: float) -> State:
        """Returns reference as State() for a multi-axis trajectory.
            The multi-axis trajectory contains single axis trajectories
            for x, y, z, and yaw. These should be derived by querying
            self.coeffs, which stores the coefficients 4x9 numpy matrix.

        Args:
            t: time at which the reference should be generated.

        Output:
            s: State() instance populated with references
        """

        s: State = State()
        t = float(t)
        for i in range(3):
            c: np.ndarray = self.coeffs[i, :]
            d1: np.ndarray = _poly_derivative_ascending(c)
            d2: np.ndarray = _poly_derivative_ascending(d1)
            d3: np.ndarray = _poly_derivative_ascending(d2)
            d4: np.ndarray = _poly_derivative_ascending(d3)
            s.pos[i, 0] = _poly_eval_ascending(c, t)
            s.vel[i, 0] = _poly_eval_ascending(d1, t)
            s.acc[i, 0] = _poly_eval_ascending(d2, t)
            s.jerk[i, 0] = _poly_eval_ascending(d3, t)
            s.snap[i, 0] = _poly_eval_ascending(d4, t)

        cy: np.ndarray = self.coeffs[3, :]
        dy1: np.ndarray = _poly_derivative_ascending(cy)
        dy2: np.ndarray = _poly_derivative_ascending(dy1)
        dy3: np.ndarray = _poly_derivative_ascending(dy2)
        s.yaw = _poly_eval_ascending(cy, t)
        s.dyaw = _poly_eval_ascending(dy1, t)
        s.d2yaw = _poly_eval_ascending(dy2, t)
        s.d3yaw = _poly_eval_ascending(dy3, t)

        yaw: float = float(s.yaw)
        cos_yaw: float = np.cos(yaw)
        sin_yaw: float = np.sin(yaw)
        s.rot = np.array(
            [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]]
        )
        return s
