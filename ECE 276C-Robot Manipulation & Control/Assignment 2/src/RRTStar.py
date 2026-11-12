import numpy as np
import math
from init_task import *

GOAL_BIAS = 0.15
MAX_ITERATIONS = 20000
EXTRA_AFTER_FIRST_PATH = 500
STEP_SIZE = np.deg2rad(15.0)


class RRTStar:

    def __init__(self, q_beg, q_end, rng=np.random.default_rng(7)):
        self.q_beg = As.wrap(q_beg)
        self.q_end = As.wrap(q_end)
        self.rng = rng
        # Directed Graph
        self.nodes = [self.q_beg.copy()]
        self.parents = [-1]
        self.costs = [0.0]
        # Path
        self.first_path = None
        self.first_iteration = None
        self.best_path = None
        self.best_path_cost = np.inf

    def nearest_index(self, q):
        return np.argmin([As.distance(n, q) for n in self.nodes])

    def steer(self, q1, q2):
        disp = As.displacement(q1, q2)
        dist = np.linalg.norm(disp)
        if dist <= STEP_SIZE:
            return As.wrap(q2)
        return As.interpolate(q1, q2, STEP_SIZE / dist)

    def trace_path(self, idx):
        path = []
        while idx != -1:
            path.append(self.nodes[idx])
            idx = self.parents[idx]
        path.reverse()
        return np.asarray(path)

    def plan(self, max_iterations=MAX_ITERATIONS):
        if not Cs.is_point_valid(self.q_beg) or not Cs.is_point_valid(self.q_end):
            return None, None, np.inf, 0
        for iteration in range(max_iterations):
            q_rand = (  # Goal bias: occasionally sample the goal directly.
                self.q_end.copy()
                if self.rng.random() < GOAL_BIAS
                else Ms.uniform_sample(self.rng)
            )
            nearest = self.nearest_index(q_rand)
            q_new = self.steer(self.nodes[nearest], q_rand)
            if not Cs.is_point_valid(q_new) or not Cs.is_edge_valid(self.nodes[nearest], q_new):
                continue
            n = len(self.nodes)
            # RRT* neighbourhood radius shrinks with tree size, with practical bounds.
            radius = float(np.clip(2.5 * math.sqrt(math.log(n + 1) / (n + 1)), 0.20, 0.85))
            near = [i for i, node in enumerate(self.nodes) if As.distance(node, q_new) <= radius]
            if nearest not in near: # if STEP_SIZE > radius, nearest may not in near
                near.append(nearest)

            parent = nearest
            best_cost_to_new = self.costs[nearest] + As.distance(self.nodes[nearest], q_new)
            for i in near:
                c = self.costs[i] + As.distance(self.nodes[i], q_new)
                if c < best_cost_to_new and Cs.is_edge_valid(self.nodes[i], q_new):
                    parent, best_cost_to_new = i, c

            new_idx = len(self.nodes)
            self.nodes.append(q_new)
            self.parents.append(parent)
            self.costs.append(best_cost_to_new)

            # Rewire neighbours through the new node when that reduces path cost.
            for i in near:
                if i == parent or i == 0:
                    continue
                candidate_cost = best_cost_to_new + As.distance(q_new, self.nodes[i])
                if candidate_cost + 1e-10 < self.costs[i] and Cs.is_edge_valid(q_new, self.nodes[i]):
                    self.parents[i] = new_idx
                    self.costs[i] = candidate_cost

            # Try connecting this node to the exact goal.
            if As.distance(q_new, self.q_end) <= STEP_SIZE and Cs.is_edge_valid(q_new, self.q_end):
                goal_cost = best_cost_to_new + As.distance(q_new, self.q_end)
                if goal_cost < self.best_path_cost:
                    self.best_path_cost = goal_cost
                    self.best_path = np.vstack([self.trace_path(new_idx), self.q_end])
                if self.first_path is None:
                    self.first_path = self.best_path.copy()
                    self.first_iteration = iteration

            # Keep running for at least 500 iterations after first feasible path.
            if self.first_path is not None and iteration >= self.first_iteration + EXTRA_AFTER_FIRST_PATH:
                return self.first_path, self.best_path, self.best_path_cost, iteration + 1
        return self.first_path, self.best_path, self.best_path_cost, max_iterations
