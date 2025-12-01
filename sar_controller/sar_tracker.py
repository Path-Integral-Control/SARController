import numpy as np
import yaml
from pathlib import Path
from types import SimpleNamespace


"""
This module performs Total Energy control designed for Sport Cub trajectory tracking.
"""

def _wrap_pi(a):
    return np.arctan2(np.sin(a), np.cos(a))

def _safe_div(x, y, eps=1e-6):
    return x / (y if abs(y) > eps else np.sign(y)*eps if y != 0.0 else eps)

class PIDController:
    def __init__(self, integral_bound, p, i, d, dt):
        self.integral_bound = integral_bound
        self.p = p
        self.i = i
        self.d = d
        self.dt = dt
        self.integral = 0
        self.last = 0

    def set_gains(self, p, i, d):
        self.p = p
        self.i = i
        self.d = d

    def compute(self, error):
        self.integral = np.clip(self.integral + error, -self.integral_bound, self.integral_bound)
        derivative = (error - self.last) / self.dt
        self.last = error
        return self.p * error + self.i * self.integral + self.d * derivative


class SARTracker:
    def __init__(self, dt, args):
        self.dt = dt
        self.g = 9.81
        #PIDController(0.3, -1.2, -0.4, -0.0, dt)
        self.elev_pid = PIDController(1, -3000, 0, 0, dt)
        self.ail_pid = PIDController(0.3, -10000, 0, 0, dt)
        self.pow_pid = PIDController(1, -1, -0.1, 0, dt)

        self.args = args #Vehicle selection
        this_file = Path(__file__).resolve()
        self.base_dir = this_file.parent / "param"

        self.reload_gains()

    def reload_gains(self):
        print(f"[SARControl] Using gain path: {self.base_dir / f"{self.args}.yaml"}")

        gain_path = self.base_dir / f"{self.args}.yaml"

        if not gain_path.exists():
            raise FileNotFoundError(f"[SARControl] Gain file not found: {gain_path}")

        with open(gain_path, "r") as f:
            raw = yaml.safe_load(f)

        self.param = SimpleNamespace(**raw)
        self.time = 0

        self.mass = self.param.mass
        self.weight = self.mass * self.g
        
        print(f"[SARControl] Gains reloaded from: {gain_path}")

    def compute_elevator(self, actual_data, ref_alt, logger=None):
        roll = actual_data['roll_est']
        V = actual_data['v_est']
        a = actual_data["a_est"]
        Q = actual_data["q_est"]
        alt_err = actual_data["z_est"] - ref_alt
        alt_tol = 3
        alt_gain = 1
        alt_bias = -np.clip(alt_err / alt_tol, -1, 1)
        q = 0.5 * self.param.rho * V * V
        L = self.weight / np.cos(roll)
        CL = L / (self.param.S * q)
        a_hat = (CL - self.param.CL0) / self.param.CLa + alt_bias * alt_gain

        de0 = (self.param.Cm0 + a * self.param.Cma + Q * self.param.Cmq) / -self.param.Cmde
        slope = q * self.param.S * self.param.cbar * self.param.Cmde / self.param.Jy
        pid = self.elev_pid.compute(a - a_hat)
        if logger is not None:
            logger(
                "Elev Info: Alpha: %0.2f: AlphaDes: %0.2f; ABias: %0.2f CL: %0.2f, Def0: %0.2f, Slope: %0.2f, PID: %0.2f, Q:%0.2f"
                % (a, a_hat, alt_bias, CL, de0, slope, pid, Q)
            )
        return  np.clip(de0 + pid / slope, -1, 1)

    def compute_aileron(self, actual_data, ref_rate, logger=None):
        roll = actual_data['roll_est']
        V = actual_data['v_est']
        P = actual_data['p_est']
        roll_hat = -np.atan(ref_rate * V / self.g)
        slope = self.param.S * self.param.span * 0.5 * self.param.rho * V * V / self.param.Jx
        pid = self.ail_pid.compute(roll - roll_hat)
        #da0 = -P * self.param.Clp / self.param.Clda
        if logger is not None:
            logger(
                "Slope: %0.2f, PID: %0.2f, Roll: %0.2f, Target: %0.2f, Ref Rate: %0.2f"
                % (slope, pid, roll, roll_hat, ref_rate)
            )
        return np.clip(pid / slope, -1, 1)

    def compute_throttle(self, actual_data, ref_vel, ref_alt, logger=None):
        alt_err = actual_data["z_est"] - ref_alt
        alt_tol = 3
        alt_gain = 2
        alt_bias = alt_gain * -np.clip(alt_err / alt_tol, -1, 1)
        V = actual_data['v_est']
        q = 0.5 * self.param.rho * V * V
        L = self.weight / np.cos(actual_data['roll_est'])
        CL = L / (self.param.S * q)
        CD = self.param.CD0 + self.param.CDCLS * CL * CL
        dt0 = q * self.param.S * CD / self.param.T_max
        slope = self.param.T_max / self.mass
        pid = self.pow_pid.compute(V - alt_bias - ref_vel)
        if logger is not None:
            logger(
                "Slope: %0.2f, PID: %0.2f, V: %0.2f, Target: %0.2f, Maintenance: %0.2f, Drag: %0.2f"
                % (slope, pid, V, ref_vel, dt0, CD * q * self.param.S)
            )
        return np.clip(pid + dt0, 0, 1)

    def compute_rudder(self, aileron):
        return np.clip(-aileron * self.param.Cnda / self.param.Cndr, -1, 1)
    
    def compute_control(self, rate, speed, alt, actual_data, logger=None):
        aileron = self.compute_aileron(actual_data, rate, logger=logger)
        elevator = self.compute_elevator(actual_data, alt, logger=logger)
        throttle = self.compute_throttle(actual_data, speed, alt, logger=logger)
        rudder = self.compute_rudder(aileron)
        return aileron, elevator, throttle, rudder