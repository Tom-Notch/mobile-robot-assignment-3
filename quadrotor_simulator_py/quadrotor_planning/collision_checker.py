#!/usr/bin/env python3

import numpy as np
from bresenham import bresenham


class CollisionChecker(object):

    def __init__(self, grid: np.ndarray) -> None:
        self.grid: np.ndarray = grid

    def has_collision(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        x0: int = int(p0[0])
        y0: int = int(p0[1])
        x1: int = int(p1[0])
        y1: int = int(p1[1])

        intersections: list[tuple[int, int]] = list(bresenham(y0, x0, y1, x1))

        for point in intersections:
            if self.grid[point[0], point[1], 0] <= 0.5:
                return True

        return False
