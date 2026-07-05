"""Minimal port of opendbc's VehicleModel (steady-state bicycle model).

Ported from opendbc/car/vehicle_model.py (comma.ai, MIT license) -- only the
curvature <-> steering-wheel-angle math needed to reconstruct the commanded
steering angle from carControl.actuators.curvature:

    commanded_angle_deg = degrees(vm.get_steer_from_curvature(-curvature, vEgo, 0.0))

(the negated curvature matches openpilot's controlsd convention).

Reference: "The Science of Vehicle Dynamics" (2014), M. Guiggiani.
"""

from __future__ import annotations

ACCELERATION_DUE_TO_GRAVITY = 9.81  # m/s^2

# carParams fields required to build the model
REQUIRED_FIELDS = (
    "mass",
    "wheelbase",
    "centerToFront",
    "steerRatio",
    "steerRatioRear",
    "tireStiffnessFront",
    "tireStiffnessRear",
)


class VehicleModel:
    def __init__(self, params: dict[str, float]):
        """params: carParams fields by name (see REQUIRED_FIELDS)."""
        self.m = float(params["mass"])
        self.l = float(params["wheelbase"])
        self.aF = float(params["centerToFront"])
        self.aR = self.l - self.aF
        self.chi = float(params.get("steerRatioRear", 0.0))
        self.cF = float(params["tireStiffnessFront"])
        self.cR = float(params["tireStiffnessRear"])
        self.sR = float(params["steerRatio"])
        if min(self.m, self.l, self.aF, self.aR, self.cF, self.cR, self.sR) <= 0:
            raise ValueError("carParams missing/zero vehicle-model fields")

    def slip_factor(self) -> float:
        return self.m * (self.cF * self.aF - self.cR * self.aR) / (
            self.l**2 * self.cF * self.cR
        )

    def curvature_factor(self, u: float) -> float:
        """Curvature per front-wheel-angle radian at speed u [m/s]."""
        sf = self.slip_factor()
        return (1.0 - self.chi) / (1.0 - sf * u**2) / self.l

    def roll_compensation(self, roll: float, u: float) -> float:
        sf = self.slip_factor()
        if abs(sf) < 1e-6:
            return 0.0
        return (ACCELERATION_DUE_TO_GRAVITY * roll) / ((1.0 / sf) - u**2)

    def calc_curvature(self, sa: float, u: float, roll: float = 0.0) -> float:
        """Curvature [1/m] for steering wheel angle sa [rad] at speed u."""
        return (self.curvature_factor(u) * sa / self.sR) + self.roll_compensation(
            roll, u
        )

    def get_steer_from_curvature(self, curv: float, u: float, roll: float = 0.0) -> float:
        """Steering wheel angle [rad] required for curvature curv [1/m]."""
        return (
            (curv - self.roll_compensation(roll, u))
            * self.sR
            / self.curvature_factor(u)
        )


def vehicle_model_from_params(params: dict[str, float] | None) -> VehicleModel | None:
    """Build a VehicleModel, or None if the carParams fields are absent/zero."""
    if not params:
        return None
    try:
        return VehicleModel(params)
    except (KeyError, ValueError, ZeroDivisionError):
        return None
