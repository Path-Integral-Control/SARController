from pathlib import Path

import yaml
import numpy as np
import numpy.linalg as la

def optimize(filename, start: np.ndarray, radius=1, gain=60):
    this_file = Path(__file__).resolve()
    file = this_file.parent / "param" / f"{filename}.yaml"
    route = yaml.load(open(file), Loader=yaml.FullLoader)
    pylons = [np.array([pylon['x'], pylon['y']]) for pylon in route['pylons']]
    gates = [[pylons[gate['p1']], pylons[gate['p2']]] for gate in route['gates']]

    #De-retardify
    gates.append(gates[0])

    def ang_diff(angle1, angle2):
        diff = angle1 - angle2
        if diff > np.pi:
            diff = -2 * np.pi + diff
        if diff < -np.pi:
            diff = 2 * np.pi + diff
        return diff

    flags = []
    #absolute position
    last = start
    for i, gate in enumerate(gates):
        #relative positions
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

    weights = [1.0 for _ in range(len(gates))]

    def calc(x, w=None):
        return flags[x] * (w if w is not None else weights[x]) + gates[x][1]

    def relative(u):
        if u >= len(gates):
            return u - len(gates) + 1
        if u < 0:
            return u + len(gates) - 1
        return u

    def presumed(y):
        ingress = calc(y) - calc(relative(y-1))
        egress = calc(relative(y+1)) - calc(y)
        vec = la.norm(egress) * ingress / la.norm(ingress) + la.norm(ingress) * egress / la.norm(egress)
        return vec / la.norm(vec)

    def evaluate(i, w):
        fore = relative(i - 1)
        aft = relative(i + 1)
        pfore = calc(fore)
        paft = calc(aft)
        pact = calc(i, w)
        baseline = paft - pfore
        offset = pact - pfore
        proj = baseline * np.dot(baseline, offset) / la.norm(baseline) ** 2
        rej = la.norm(offset - proj)
        pro = la.norm(proj)
        drej = np.dot(flags[i], la.norm(rej))

        vfore = presumed(fore)
        vaft = presumed(aft)
        vme = presumed(i)
        afore = np.arctan2(vfore[1], vfore[0])
        aaft = np.arctan2(vaft[1], vaft[0])
        ame = np.arctan2(vme[1], vme[0])

        difffore = abs(ang_diff(afore, ame))
        diffaft = abs(ang_diff(aaft, ame))
        total = difffore / rej + diffaft / rej + difffore / pro + diffaft / pro

        distance = la.norm(pfore - pact) + la.norm(paft - pact)
        return distance + gain * total ** 2

    radius = 1
    step = 0.3
    for _ in range(1000):
        for i in range(len(flags)):
            low = max(weights[i] - step, radius)
            less = evaluate(i, low)
            more = evaluate(i, weights[i] + step)
            if less < more:
                weights[i] = low
            else:
                weights[i] += step

    return [gate[1] + flag * w for gate, flag, w in zip(gates, flags, weights)], pylons