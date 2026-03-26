#!/usr/bin/env python3

import numpy as np

from quadrotor_simulator_py.quadrotor_planning.polynomialtrajectory import (
    PolynomialTrajectory,
)


def _mueller_constraint_matrix(T: float) -> np.ndarray:
    """6x6 matrix A with A @ c = [p0, v0, a0, pf, vf, af] for p(t)=sum c_i t^i."""
    A = np.zeros((6, 6))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 2.0
    Tp = np.array([T**i for i in range(6)])
    for i in range(6):
        A[3, i] = Tp[i]
    for i in range(1, 6):
        A[4, i] = i * T ** (i - 1)
    for i in range(2, 6):
        A[5, i] = i * (i - 1) * T ** (i - 2)
    return A


class MuellerTrajectory(PolynomialTrajectory):
    def __init__(self, x0: np.ndarray, xf: np.ndarray, T: float) -> None:
        """Calculates the coefficients of the 5th order
        polynomial trajectory given fixed starting position,
        velocity, and acceleration as well as duration, T.
        The coefficients should be stored as self.coefficients.

        The polynomial is written as c0 + c1*t + ... c5*t^5.
        The input to the function are constraints and duration.
        The output of the function are coefficients for a single-axis
        polynomial trajectory.

        Reference: https://doi.org/10.1109/TRO.2015.2479878

        Args:
            x0: 1x3 numpy array consisting of starting position, velocity, and acceleration
            xf: 1x3 numpy array consisting of position, velocity, and acceleration at time T
            T: duration of the trajectory

        Output:
            coefficients: 1x6 numpy array of coefficients
        """

        self.T: float = T
        x0_flat: np.ndarray = np.asarray(x0).flatten()
        xf_flat: np.ndarray = np.asarray(xf).flatten()
        p0: float = float(x0_flat[0])
        v0: float = float(x0_flat[1])
        a0: float = float(x0_flat[2])
        pf: float = float(xf_flat[0])
        vf: float = float(xf_flat[1])
        af: float = float(xf_flat[2])
        b: np.ndarray = np.array([p0, v0, a0, pf, vf, af])
        A: np.ndarray = _mueller_constraint_matrix(T)
        self.coefficients: np.ndarray = np.linalg.solve(A, b)
