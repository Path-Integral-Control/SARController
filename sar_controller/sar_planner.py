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
    def __init__(self, dt, waypoints, start, tolerance=20, alt=7.5):
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
        self.failed = False
        self.tolerance = tolerance
        self.alt = alt
        self.integral = 0


    def failsafe(self, pos, vel, logger=None):
        current = self.waypoints[self.idx]
        last = self.waypoints[self.idx - 1]
        path = current - last
        offset = pos - current

        if np.dot(path, offset) > 0:
            self.failed = False
            return self.plan(pos, vel, logger)

        target = np.arctan2(path[1], path[0])
        heading = np.arctan2(vel[1], vel[0])
        bearing = np.arctan2(offset[1], offset[0])
        btdiv = ang_diff(bearing, target)
        ideal = 2 * btdiv + target
        if ideal > np.pi:
            ideal = -2 * np.pi + ideal
        elif ideal < -np.pi:
            ideal = 2 * np.pi + ideal

        control = ang_diff(ideal, heading)

        return control, 10, self.alt

        proj = path * np.dot(path, offset) / la.norm(path)**2
        rej = la.norm(offset - proj)





    def fail(self, pos, vel, logger=None):
        best = self.idx
        dist = 0
        for i in range(len(self.waypoints)):
            attempt = la.norm(self.waypoints[i] - pos)
            if attempt > dist:
                dist = attempt
                best = i

        self.idx = best
        self.failed = True
        if logger is not None:
            logger(f"Failed: New Waypoint {self.waypoints[self.idx]}, Dir {self.waypoints[self.idx] - self.waypoints[self.idx - 1]}, Pos: {pos}")


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

        # Activate failsafe controller if necessary
        if self.failed:
            return self.failsafe(pos, vel, logger)

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

        # Activate failsafe mode if necessary
        safety = abs(cross) + abs(theta) * 10


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

        if safety > self.tolerance:
            self.fail(pos, vel, logger)
            return self.failsafe(pos, vel, logger)
        p = 0.05
        d = 1.1

        # Control is default control plus PID adjustment
        control = np.clip(-p * cross + -d * np.sin(theta) + default, -2, 2)

        # Simple deterministic airspeed control from turn radius
        return control, 0.4 * abs(dist / delta) + 6, self.alt


    # Get current waypoint index
    def get_index(self):
        return self.idx
