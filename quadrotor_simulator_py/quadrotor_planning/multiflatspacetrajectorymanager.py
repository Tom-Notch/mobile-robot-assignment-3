#!/usr/bin/env python3

from typing import List

import numpy as np

from quadrotor_simulator_py.quadrotor_control.state import State
from quadrotor_simulator_py.quadrotor_planning.flatspacetrajectory import (
    FlatSpaceTrajectory,
)


class MultiFlatSpaceTrajectoryManager(object):
    def __init__(self) -> None:
        """
        initializes empty list of FlatSpaceTrajectory (self.trajs)
        initializes empty array of knot times (self.knotts)
        """
        self.trajs: List[FlatSpaceTrajectory] = []
        self.knotts: np.ndarray = np.array([0.0])

    def append(self, traj: FlatSpaceTrajectory, t: float) -> None:
        """Appends a FlatSpaceTrajectory to list of FlatSpaceTrajectory's
            and manages knotts appropriatesly
        Args:
            traj: FlatSpaceTrajectory
            t: duration of FlatSpaceTrajectory
        """
        self.trajs.append(traj)
        self.knotts = np.append(self.knotts, self.knotts[-1] + t)

    def get_ref(self, t: float) -> State:
        """Gets reference at time t, where time may be between 0
        and last knot time.

        Args:
           t: time between 0 and self.knotts[-1]

        Output:
           ref: State() queried at time t
        """

        if not self.trajs:
            return State()
        t = float(t)
        if t <= self.knotts[0]:
            return self.trajs[0].get_ref(0.0)
        if t >= self.knotts[-1]:
            last: FlatSpaceTrajectory = self.trajs[-1]
            return last.get_ref(last.T)
        idx: int = int(np.searchsorted(self.knotts, t, side="right") - 1)
        idx = max(0, min(idx, len(self.trajs) - 1))
        tau: float = t - self.knotts[idx]
        seg: FlatSpaceTrajectory = self.trajs[idx]
        if tau > seg.T:
            tau = seg.T
        return seg.get_ref(tau)
