#!/usr/bin/env python3

from typing import List

import numpy as np

from quadrotor_simulator_py.quadrotor_control.state import State
from quadrotor_simulator_py.quadrotor_planning.muellertrajectory import (
    MuellerTrajectory,
)


class FlatSpaceTrajectory(object):

    def __init__(self, s0: State, sf: State, T: float) -> None:
        self.T: float = T
        self.trajs: List[MuellerTrajectory] = []

        x0s: np.ndarray = self.state_to_arrays(s0)
        xfs: np.ndarray = self.state_to_arrays(sf)
        if len(x0s) == 4 and len(xfs) == 4:
            for i in range(4):
                self.trajs.append(MuellerTrajectory(x0s[i, :], xfs[i, :], T))
        else:
            raise Exception("FlatSpaceTrajectory Failure! len(x0s) == xfs == 4")

    def state_to_arrays(self, s: State) -> np.ndarray:
        xs: np.ndarray = np.zeros((4, 3))
        for i in range(3):
            xs[i, :] = np.array([s.pos[i, 0], s.vel[i, 0], s.acc[i, 0]])
        xs[3, :] = np.array([s.yaw, s.dyaw, s.d2yaw])
        return xs

    def get_ref(self, t: float) -> State:
        """Returns reference as State() for a multi-axis trajectory.
            The multi-axis trajectory contains single axis trajectories
            for x, y, z, and yaw. These should be derived by querying
            self.trajs, which stores the coefficients as a list of
            1xn numpy arrays.

        Args:
            t: time at which the reference should be generated.

        Output:
            s: State() instance populated with references
        """

        s: State = State()
        for i in range(3):
            g: np.ndarray = self.trajs[i].get_ref(t, 2)
            s.pos[i, 0] = g[0]
            s.vel[i, 0] = g[1]
            s.acc[i, 0] = g[2]
        gy: np.ndarray = self.trajs[3].get_ref(t, 2)
        s.yaw = float(gy[0])
        s.dyaw = float(gy[1])
        s.d2yaw = float(gy[2])
        return s
