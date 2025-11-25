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
        self.idx = start
        self.gate_idx = 1
        self.rate = 0
        self.waypoints = [np.array(pt[0:2]) for pt in waypoints]
        self.gates = self.init_gates(self.waypoints, pos[0:2])
        for gate in self.gates:
            print(gate)

    def init_gates(self, waypoints, pos):
        gates = [pos]
        for i, waypoint in enumerate(waypoints):
            app = waypoint - waypoints[i - 1]
            if np.linalg.norm(app) < 1:
                continue
            dep = waypoints[(i + 1) % len(waypoints)] - waypoint
            appl = np.linalg.norm(app) + 0.01
            depl = np.linalg.norm(dep) + 0.01
            gates.append(waypoint + (app / appl - dep / depl) * 2)
        print(gates)
        return gates


    def plan(self, pos, vel, logger=None):
        gate = self.gates[self.gate_idx]
        waypoint = self.waypoints[self.idx]
        wlast = self.waypoints[self.idx - 1]
        wpath = waypoint - wlast
        woffset = pos - waypoint
        if np.dot(wpath, woffset) > 0:
            self.idx += 1
            if logger is not None:
                logger(f"[SARPlanner] Passed Waypoint, Pos: {pos}, Vel: {vel}, Waypoint: {waypoint}")
        glast = self.gates[self.gate_idx - 1]
        gpath = gate - glast
        goffset = pos - gate
        if np.dot(gpath, goffset) > 0:
            self.gate_idx += 1
            if logger is not None:
                logger(f"[SARPlanner] Passed Gate, Pos: {pos}, Vel: {vel}, Gate: {gate}")

            if self.gate_idx == len(self.gates):
                self.gate_idx = 0
            gate = self.gates[self.gate_idx]

        """
        rate, speed, alt = (
            self.find_curve(
                pos, vel, gate, logger=logger
            )
        )
        """
        """
        heading = np.arctan2(vel[1], vel[0])
        bearing = np.arctan2(gate.pos[1] - pos[1], gate.pos[0] - pos[0])
        final = np.arctan2(gate.vec[1], gate.vec[0])
        bhdiv = ang_diff(heading, bearing)
        bfdiv = ang_diff(bearing, final)
        divdiv = ang_diff(bhdiv, bfdiv)
        target = bfdiv + bearing
        if target > np.pi:
            target -= 2 * np.pi

        if target < -np.pi:
            target += 2 * np.pi

        distance = np.linalg.norm(pos - gate.pos)
        control = ang_diff(target, heading)
        if logger is not None:
            pass#logger(f"Bearing: {round(bearing, 3)}, Heading: {round(heading, 3)}, Final: {round(final, 3)}, BHDiv: {round(bhdiv, 3)}, BFDiv: {round(bfdiv, 3)}, Target: {round(target, 3)}, Control: {round(control, 3)}, DivDiv: {round(divdiv, 3)}")
        """
        target = gate
        heading = np.arctan2(vel[1], vel[0])
        bearing = np.arctan2(target[1] - pos[1], target[0] - pos[0])
        distance = np.linalg.norm(pos - target)
        speed = np.linalg.norm(vel)

        control = 1.1 * speed * ang_diff(bearing, heading) / distance

        if logger is not None:
            pass#logger(f"Bearing: {round(bearing, 3)}, Heading: {round(heading, 3)}, Control: {round(control, 3)}")

        return np.clip(control, -3, 3), 20, 7.5


    def get_index(self):
        return self.idx
