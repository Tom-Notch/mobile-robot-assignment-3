#!/usr/bin/env python3

from typing import Dict, List, Optional, Tuple

import numpy as np

from quadrotor_simulator_py.quadrotor_planning.collision_checker import CollisionChecker

GRID_WIDTH = 300

STEP = 12.0
GOAL_THRESH = 18.0
MAX_ITERS = 50000


class Node(object):

    def __init__(self, parent_idx: int, position: np.ndarray) -> None:
        self.parent_idx: int = parent_idx
        self.idx: int = int(position[1] * GRID_WIDTH + position[0])
        self.position: np.ndarray = position
        self.children: List[int] = []

    def append_child(self, child: "Node") -> None:
        self.children.append(child.idx)


class RRT(object):

    def __init__(self, start: np.ndarray, end: np.ndarray, grid: np.ndarray) -> None:
        self.collision_checker: CollisionChecker = CollisionChecker(grid)
        self.start: np.ndarray = start
        self.end: np.ndarray = end
        self.grid: np.ndarray = grid
        self.tree: Dict[int, Node] = {}
        self.insert_root_node(start)

    def run(self) -> Node:
        """Runs the RRT algorithm until a path is found between the start and
            end positions.

        Output:
            node: node representing end position
        """

        end_i: np.ndarray = np.round(self.end).astype(int)
        for _ in range(MAX_ITERS):
            _idx: int
            sample_pos: np.ndarray
            _idx, sample_pos = self.sample()
            nearest: Node = self.find_nearest_node(sample_pos)
            new_pos: np.ndarray = self._steer(nearest.position, sample_pos)
            new_idx: int = self.get_idx(new_pos)
            if new_idx in self.tree:
                continue
            nearest_int: np.ndarray = np.round(nearest.position).astype(int)
            if self.collision_checker.has_collision(new_pos, nearest_int):
                continue
            child: Node = self.insert_node(nearest, new_pos)
            if (
                np.linalg.norm(new_pos.astype(float) - self.end.astype(float))
                < GOAL_THRESH
            ):
                end_idx: int = self.get_idx(end_i)
                if (
                    end_idx not in self.tree
                    and not self.collision_checker.has_collision(end_i, child.position)
                ):
                    return self.insert_node(child, end_i)
        fail: Node = Node(self.get_idx(self.start), np.array(self.start))
        fail.parent_idx = fail.idx
        return fail

    def _steer(self, from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
        d: np.ndarray = to_pos.astype(float) - from_pos.astype(float)
        dist: float = float(np.linalg.norm(d))
        if dist < 1e-9:
            return np.round(from_pos).astype(int)
        if dist <= STEP:
            return np.round(to_pos).astype(int)
        nxt: np.ndarray = from_pos.astype(float) + STEP * (d / dist)
        return np.round(nxt).astype(int)

    def get_idx(self, position: np.ndarray) -> int:
        x: float = float(position[0])
        y: float = float(position[1])
        return int(y * GRID_WIDTH + x)

    def insert_root_node(self, position: np.ndarray) -> None:
        idx: int = self.get_idx(position)
        root: Node = Node(idx, position)
        self.tree[idx] = root

    def find_nearest_node(self, position: np.ndarray) -> Node:
        """Finds the node that has a position closest to the input position

        Args:
            position: 1x2 numpy array

        Output:
            node: Node class instance representing node with
                  closest position to input.
        """

        best: Optional[Node] = None
        best_d: float = float("inf")
        for node in self.tree.values():
            d: float = float(np.linalg.norm(node.position - position))
            if d < best_d:
                best_d = d
                best = node
        assert best is not None
        return best

    def insert_node(self, parent_node: Node, position: np.ndarray) -> Node:
        """Create child node given parent node and position of child node

        Args:
            position: 1x2 numpy array
            parent_node: Node instance that represents parent

        Output:
            child_node: Node class instance representing child node
        """

        idx: int = self.get_idx(position)
        child: Node = Node(parent_node.idx, position)
        self.tree[idx] = child
        parent_node.append_child(child)
        return child

    def sample(self) -> Tuple[int, np.ndarray]:
        """Randomly sample from list of existing positions within the
            grid

        Output:
            (idx, position): integer index and position pair
                             position represents a 1x2 numpy array
        """

        h: int = self.grid.shape[0]
        w: int = self.grid.shape[1]
        x: int = int(np.random.randint(0, w))
        y: int = int(np.random.randint(0, h))
        pos: np.ndarray = np.array([x, y], dtype=float)
        return self.get_idx(pos), pos
