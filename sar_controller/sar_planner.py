import numpy as np
from random import random

######################
# Waypoint Target Algorithm with cross-track flow field and Alongtrack switching modes
######################


def angle_rad_wrapper(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def proj(pos, point, vector):
    poff = pos - point
    return np.dot(poff, vector)

def ang_diff(angle1, angle2):
    diff = angle1 - angle2
    if diff > np.pi:
        diff = -2 * np.pi + diff
    if diff < -np.pi:
        diff = 2 * np.pi + diff
    return diff


class SARPlanner:
    def __init__(self, dt, waypoints, start, pos):
        self.dt = dt
        self.idx = start + 1
        self.gate_idx = 1
        self.rate = 0
        self.waypoints = [start] + [np.array(pt[0:2]) for pt in waypoints]


    def plan(self, pos, vel, logger=None):
        waypoint = self.waypoints[self.idx]
        wlast = self.waypoints[self.idx - 1]
        wpath = waypoint - wlast
        woffset = pos - waypoint
        logger(f"[SARPlanner] Finding Waypoint, Pos: {pos}, Vel: {vel}, Waypoint: {waypoint}")
        if np.dot(wpath, woffset) > 0:
            self.idx += 1
            if logger is not None:
                logger(f"[SARPlanner] Passed Waypoint, Pos: {pos}, Vel: {vel}, Waypoint: {waypoint}")

        upper_gain = 1
        lower_gain = 1.1

        target = waypoint
        heading = np.arctan2(vel[1], vel[0])
        bearing = np.arctan2(target[1] - pos[1], target[0] - pos[0])
        distance = np.linalg.norm(pos - target)
        speed = np.linalg.norm(vel)

        control = np.clip(upper_gain * speed * ang_diff(bearing, heading) / distance**lower_gain, -3, 3)

        if abs(control) > abs(self.rate):
            self.rate = 0.95 * self.rate + 0.05 * control
        else:
            self.rate = control

        if logger is not None:
            pass#logger(f"Bearing: {round(bearing, 3)}, Heading: {round(heading, 3)}, Control: {round(control, 3)}")

        return np.clip(control, -3, 3), 20, 7.5


    def get_index(self):
        return self.idx
