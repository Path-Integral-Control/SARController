import numpy as np
import numpy.linalg as la


# Difference between two angles
def ang_diff(angle1, angle2):
    diff = angle1 - angle2
    if diff > np.pi:
        diff = -2 * np.pi + diff
    if diff < -np.pi:
        diff = 2 * np.pi + diff
    return diff


class SARPlanner:
    def __init__(self, dt, waypoints, start):
        """
        Parameters
        ----------
        dt - timestep
        waypoints - waypoint list
        start - index of first waypoint of real trajectory (disincluding lead-in trajectory)
        """
        self.dt = dt
        self.start = start
        self.idx = 1
        self.rate = 0
        self.waypoints = [np.array(pt[0:2]) for pt in waypoints]
        self.starting = True


    # Copy-pasted from optimizer.py
    def relative(self, u):
        if u >= len(self.waypoints):
            return u - len(self.waypoints) + 1
        if u < 0:
            return u + len(self.waypoints) - 1
        return u


    # Copy-pasted from optimizer.py
    def presumed(self, y):
        ingress = self.waypoints[y] - self.waypoints[self.relative(y - 1)]
        egress = self.waypoints[self.relative(y + 1)] - self.waypoints[y]
        vec = la.norm(egress) * ingress / la.norm(ingress) + la.norm(ingress) * egress / la.norm(egress)
        return vec / la.norm(vec)


    def plan(self, pos, vel, logger=None):
        """
        Find turn rate, airspeed, and altitude from position and velocity

        Parameters
        ----------
        pos - current position
        vel - current velocity
        logger - logger callback

        Returns
        -------
        Turn rate
        Airspeed
        Altitude
        """

        # Fetch current waypoint
        waypoint = self.waypoints[self.idx]

        # Determine whether waypoint has been passed w/ vector projection
        wlast = self.waypoints[self.idx - 1]
        wpath = waypoint - wlast
        woffset = pos - waypoint

        # Loop to skip multiple waypoints at once if necessary
        while np.dot(wpath, woffset) > 0:
            self.idx += 1
            if self.idx == len(self.waypoints):
                self.idx = self.start

            if self.idx > self.start:
                self.starting = False

            # Reset waypoint data
            waypoint = self.waypoints[self.idx]
            wlast = self.waypoints[self.idx - 1]
            wpath = waypoint - wlast
            woffset = pos - waypoint

            if logger is not None:
                logger(f"[SARPlanner] Passed Waypoint, Pos: {pos}, Vel: {vel}, Waypoint: {waypoint}")

        # Find cross-track error
        path = wpath
        offset = pos - wlast
        projection = path * np.dot(path, offset) / la.norm(path)**2
        reject = offset - projection
        cross = la.norm(reject) * -np.sign(reject[0] * path[1] - path[0] * reject[1])

        # Find heading error
        pangle = np.arctan2(path[1], path[0])
        heading = np.arctan2(vel[1], vel[0])
        theta = ang_diff(heading, pangle)

        # Fetch next waypoint
        wnext = self.waypoints[self.idx + 1] if self.idx + 1 < len(self.waypoints) else waypoint * 2 - wlast
        npath = wnext - waypoint

        # Find distance of current turn
        dist = (la.norm(npath) + la.norm(path)) / 2

        # Find default turn rate from current and next control segments
        inangle = np.arctan2(path[1], path[0])
        outangle = np.arctan2(npath[1], npath[0])
        delta = ang_diff(outangle, inangle)
        default = la.norm(vel) * delta / dist

        # PID control for cross-track error using idea that derivtive of cross-track is sin(theta)
        p = 0.012
        d = 2.3

        if self.starting:
            default = 0
            p = 0.1
            d = 3



        # Control is default control plus PID adjustment
        control = -p * cross + -d * np.sin(theta) + default

        # TODO: add actual airspeed control if necessary
        return np.clip(control, -3, 3), 20, 7.5


    # Get current waypoint index
    def get_index(self):
        return self.idx
