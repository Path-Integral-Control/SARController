from pathlib import Path

import yaml
import numpy as np
import numpy.linalg as la
from scipy.interpolate import make_interp_spline


def optimize(filename, start: np.ndarray, radius=1, gain=60, offset=np.array([0, 0])):
    """
    Builds an optimized reference trajectory from a route description file

    Parameters
    ----------
    filename - route description YAML file
    start - expected start location
    svec - direction expected to start controlled flight at
    ppos - location to stage at before starting course
    pvec - partial vector, the direction to go at the intermediate staging point before the course
    radius - minimum turn radius around each pylon
    gain - relative gain of curvature cost versus distance cost

    Returns
    -------
    waypoints - ordered list of numpy coordinate waypoints
    pylons - locations of original pylons
    """

    # Read the course file
    this_file = Path(__file__).resolve()
    file = this_file.parent / "param" / f"{filename}.yaml"
    route = yaml.load(open(file), Loader=yaml.FullLoader)
    pylons = [np.array([pylon['x'], pylon['y']]) + offset for pylon in route['pylons']]
    gates = [[pylons[gate['p1']], pylons[gate['p2']]] for gate in route['gates']]
    gates.append(gates[0])


    # Calculate the difference between two angles
    def ang_diff(angle1, angle2):
        diff = angle1 - angle2
        if diff > np.pi:
            diff = -2 * np.pi + diff
        if diff < -np.pi:
            diff = 2 * np.pi + diff
        return diff

    # Determine the side of each pylon to fly
    # Flags are vectors relative to each pylon
    # Flags will later be weighted and added to their pylon loctions to create the primary waypoints
    flags = []
    last = start

    # Loop through gates and alternate the side of each pylon to place the flag
    for i, gate in enumerate(gates):
        offset = last - gate[0]
        vector = gate[1] - gate[0]
        proj = vector * np.dot(vector, offset) / la.norm(vector)**2
        initial = 5 * -(offset - proj)

        nex = gates[(i + 1) % len(gates) if i < len(gates) - 1 else 1]
        nexvec = nex[0] - nex[1]
        spline = None
        if np.dot(vector + nexvec, initial) < 0:
            spline = -(nexvec + vector)
        else:
            spline = nexvec * la.norm(vector) / la.norm(nexvec) + vector * la.norm(nexvec) / la.norm(vector)

        flag = spline / la.norm(spline)

        last = flag + gate[1]
        flags.append(flag)

    # Weights are the distances to multiply each flag vector by to create the offset primary waypoints
    weights = [1.0 for _ in range(len(gates))]

    # Calculate the location of a given primary waypoint with an optional test weight
    # If not given, use the existing weight
    def calc(x, w=None):
        return flags[x] * (w if w is not None else weights[x]) + gates[x][1]

    # Ensures safe indexing
    def relative(u):
        if u >= len(gates):
            return u - len(gates) + 1
        if u < 0:
            return u + len(gates) - 1
        return u

    # Calculate the presumed direction vector at a given primary waypoint, based on its neighbors
    def presumed(y):
        ingress = calc(y) - calc(relative(y-1))
        egress = calc(relative(y+1)) - calc(y)
        vec = la.norm(egress) * ingress / la.norm(ingress) + la.norm(ingress) * egress / la.norm(egress)
        return vec / la.norm(vec)


    def evaluate(i, w):
        """
        Evaluate the cost of a given pylon with a given weight

        Parameters
        ----------
        i - index of primary pylon
        w - weight to evaluate for

        Returns
        -------
        Calculated cost
        """

        """
        This entire section uses a heuistic to calculate the average amount of curvature in between two pylons.
        The general idea is to calculate the amount of distance to turn before an overshoot and the change in heading 
        required. The heading change divided by the distance is the total curvature. This assumes no overshoots.
        
        Distance required
        """

        # Fetch all relevent positions and presumed directions, using fore and aft terminology
        fore = relative(i - 1)
        aft = relative(i + 1)
        pfore = calc(fore)
        paft = calc(aft)
        pact = calc(i, w)
        vfore = presumed(fore)
        vaft = presumed(aft)
        vme = presumed(i)

        # Line directly connecting previous and next points
        baseline = paft - pfore

        # Line from previous to next points
        offset = pact - pfore

        # Projections and rejections of pylon position onto the baseline
        proj = baseline * np.dot(baseline, offset) / la.norm(baseline) ** 2
        rej = la.norm(offset - proj)
        pro = la.norm(proj)

        # Find heading angles at each point
        afore = np.arctan2(vfore[1], vfore[0])
        aaft = np.arctan2(vaft[1], vaft[0])
        ame = np.arctan2(vme[1], vme[0])

        # Total heading change in each direction
        difffore = abs(ang_diff(afore, ame))
        diffaft = abs(ang_diff(aaft, ame))

        """
        Assume entire turn occurs in either projection or rejection spaces. We simply add these two to get a heuristic 
        of total curvature assuming no overshoots. Will be off by a roughly constant factor from true value, but
        constant is simply absorbed into the curvature gain.
        """
        if rej < pro:
            total = difffore / rej + diffaft / rej
        else:
            total = difffore / pro + diffaft / pro

        if total > 0.35:
            total *= 3

        # Apply a penalty for close pylon approaches
        radius_gain = 0

        # Calculate total distance as direct distance
        # Not strictly true obviously, but error is proportional to curvature, which is also penalized
        distance = la.norm(pfore - pact) + la.norm(paft - pact)

        # Penalize distance linearly and curvature quadratically (curvature reduces maximum speed roughly quadratically)
        return distance + gain * total ** 2 + radius_gain / w


    # Stochastic gradient descent to improve reference trajectory, initialized at min-radius turns
    step = 0.1
    for _ in range(1000):
        for i in range(len(flags)):

            # Test higher and lower weights and select better option
            # I was lazy and didn't want to calculate gradients
            low = max(weights[i] - step, radius)
            less = evaluate(i, low)
            more = evaluate(i, weights[i] + step)
            if less < more:
                weights[i] = low
            else:
                weights[i] += step

    pts_list = [gates[i][1] + flags[i] * weights[i] for i in list(range(1, len(flags)))]
    pts_list.append(pts_list[0])
    pts = np.array(pts_list).transpose()
    dp = pts[:, 1:] - pts[:, :-1]
    l = (dp ** 2).sum(axis=0)
    u_cord = np.sqrt(l).cumsum()
    u_cord = np.r_[0, u_cord]

    spl = make_interp_spline(u_cord, pts, axis=1, bc_type="periodic", k=3)  # note p is a 2D array
    uu = np.linspace(u_cord[0], u_cord[-1], 51)
    xx, yy = spl(uu)
    waypoints = list(np.stack([xx, yy]).transpose())

    # Return both the waypoints and the original pylons
    return waypoints, pylons, 10