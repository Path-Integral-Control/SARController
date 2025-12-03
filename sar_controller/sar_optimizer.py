from pathlib import Path

import yaml
import numpy as np
import numpy.linalg as la

def optimize(filename, start: np.ndarray, svec: np.ndarray, ppos: np.ndarray, pvec: np.ndarray, radius=1, gain=60):
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
    pylons = [np.array([pylon['x'], pylon['y']]) for pylon in route['pylons']]
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
        total = difffore / rej + diffaft / rej + difffore / pro + diffaft / pro

        # Apply a penalty for close pylon approaches
        radius_gain = 0

        # Calculate total distance as direct distance
        # Not strictly true obviously, but error is proportional to curvature, which is also penalized
        distance = la.norm(pfore - pact) + la.norm(paft - pact)

        # Penalize distance linearly and curvature quadratically (curvature reduces maximum speed roughly quadratically)
        return distance + gain * total ** 2 + radius_gain / w

    def find_intermediates(forep, forev, pos, vel):
        """
        Find intermediate waypoints between two primary waypoints

        Parameters
        ----------
        forep - position at first waypoint
        forev - direction at first waypoint
        pos - position at second waypoint
        vel - direction at second waypoint

        Returns
        -------
        List of intermediate waypoints
        """

        # Direction angles at both primaries
        fa = np.arctan2(forev[1], forev[0])
        ca = np.arctan2(vel[1], vel[0])

        # Angular dfference between directions
        darc = ang_diff(ca, fa)

        # Calculate vectors pointing towards turn circle center from both points
        off = np.sign(darc) * np.pi / 2
        fina = off + fa
        cina = off + ca
        finv = np.array([np.cos(fina), np.sin(fina)])
        cinv = np.array([np.cos(cina), np.sin(cina)])

        # If turn circle is infinite (straight line segment), just return evenly spaced points
        if np.dot(finv, cinv) / (la.norm(cinv) * la.norm(finv)) == 1:
            return [pt for pt in np.linspace(forep, pos, 10, endpoint=False)]

        """
        Calculate seperate circles for both points to interpolate between.
        Uses vector projection to calculate circle radius, then uses radius to find center.
        """
        rej_b = finv - cinv * np.dot(finv, cinv) / la.norm(cinv)
        err_b = pos - forep
        rad_b = np.dot(err_b, rej_b) / la.norm(rej_b) ** 2
        center_b = rad_b * cinv + pos

        rej_f = cinv - finv * np.dot(finv, cinv) / la.norm(finv)
        err_f = forep - pos
        rad_f = np.dot(err_f, rej_f) / la.norm(rej_f) ** 2
        center_f = rad_f * finv + forep

        # Place points along interpolated turn ellipse
        pts = []
        granules = 10
        for i in range(granules):

            # Evenly angularly-spaced points
            theta = fina + i * darc / granules
            vec = np.array([np.cos(theta), np.sin(theta)])

            # Linearly interpolate between the two turn circles
            ratio = i / granules
            pts.append(ratio * (center_b - rad_b * vec) + (1 - ratio) * (center_f - rad_f * vec))

        return pts

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

    # Manually add lead in path to course start from start to staging point to first primary waypoint
    partp = ppos
    partv = pvec
    partv = partv / la.norm(partv)

    points = find_intermediates(start, svec, partp, partv)
    points += find_intermediates(partp, partv, calc(0), presumed(0))

    # Add intermediate points between each primary waypoint
    for i in range(1, len(flags)):
        curr = calc(i)
        fore = calc(relative(i - 1))
        cv = presumed(i)
        fv = presumed(relative(i - 1))
        points += find_intermediates(fore, fv, curr, cv)

    # Return both the waypoints and the original pylons
    return points, pylons