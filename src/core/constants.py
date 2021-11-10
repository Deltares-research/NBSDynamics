from src.core.base_model import BaseModel
from pydantic import root_validator, validator
from typing import Optional
from pathlib import Path
import numpy as np


class Constants(BaseModel):
    """Object containing all constants used in coral_model simulations."""

    # Input file
    input_file: Optional[Path]

    # Processes
    fme: bool = False
    tme: bool = False
    pfd: bool = False
    warn_proc: bool = True

    # light micro-environment
    Kd0: float = 0.1
    theta_max: float = 0.5

    # flow micro-environment
    Cs: float = 0.17
    Cm: float = 1.7
    Cf: float = 0.01
    nu: float = 1e-6
    alpha: float = 1e-7
    psi: float = 2
    wcAngle: float = 0.0
    rd: float = 500
    numericTheta: float = 0.5
    err: float = 1e-3
    maxiter_k: int = 1e5
    maxiter_aw: int = 1e5

    # thermal micro-environment
    K0: float = 80.0
    ap: float = 0.4
    k: float = 0.6089

    # photosynthetic light dependency
    iota: float = 0.6
    ik_max: float = 372.32
    pm_max: float = 1.0
    betaI: float = 0.34
    betaP: float = 0.09
    Icomp: float = 0.01

    # photosynthetic thermal dependency
    Ea: float = 6e4
    R: float = 8.31446261815324
    k_var: float = 2.45
    nn: float = 60

    # photosynthetic flow dependency
    pfd_min: float = 0.68886964
    ucr: float = 0.5173

    # population dynamics
    r_growth: float = 0.002
    r_recovery: float = 0.2
    r_mortality: float = 0.04
    r_bleaching: float = 8.0

    # calcification
    gC: float = 0.5
    omegaA0: float = 5.0
    omega0: float = 0.14587415
    kappaA: float = 0.66236107

    # morphological development
    rf: float = 1.0
    rp: float = 1.0
    prop_form: float = 0.1
    prop_plate: float = 0.5
    prop_plate_flow: float = 0.1
    prop_space: float = 0.5 / np.sqrt(2.0)
    prop_space_light: float = 0.1
    prop_space_flow: float = 0.1
    u0: float = 0.2
    rho_c: float = 1600.0

    # dislodgement criterion
    sigma_t: float = 2e5
    Cd: float = 1.0
    rho_w: float = 1025.0

    # coral recruitment
    no_larvae: float = 1e6
    prob_settle: float = 1e-4
    d_larvae: float = 1e-3

    @validator("maxiter_k", "maxiter_aw", pre=True, always=True)
    @classmethod
    def validate_scientific_int_value(cls, v) -> int:
        """
        Validates the parameters that can be provided with scientific notation.

        Args:
            v (Any): Scientific value to validate.

        Returns:
            int: Validated value as integer.
        """
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str):
            return int(float(v))

        raise NotImplementedError(f"No converter available for {type(v)}.")

    @root_validator
    @classmethod
    def check_processes(cls, values: dict) -> dict:
        """
        Validates the input values so that the processes are compatible between themselves.

        Args:
            values (dict): Dictionary of values already validated individually.

        Returns:
            dict: Dictionary of validated values as a whole.
        """
        if not values["pfd"]:
            if values["fme"] and values["warn_proc"]:
                print(
                    "WARNING: Flow micro-environment (FME) not possible "
                    "when photosynthetic flow dependency (PFD) is disabled."
                )
            values["fme"] = False
            values["tme"] = False

        else:
            if not values["fme"]:
                if values["tme"] and values["warn_proc"]:
                    print(
                        "WARNING: Thermal micro-environment (TME) not possible "
                        "when flow micro-environment is disabled."
                    )
                values["tme"] = False

        if values["tme"] and values["warn_proc"]:
            print("WARNING: Thermal micro-environment not fully implemented yet.")

        if not values["pfd"] and values["warn_proc"]:
            print(
                "WARNING: Exclusion of photosynthetic flow dependency not fully implemented yet."
            )
        return values

    @classmethod
    def from_input_file(cls, input_file: Path):
        """
        Generates a 'Constants' class based on the defined parameters in the input_file.

        Args:
            input_file (Path): Path to the constants input (.txt) file.
        """

        def split_line(line: str):
            s_line = line.split("=")
            if len(s_line) <= 1:
                raise ValueError
            return s_line[0].strip(), s_line[1].strip()

        def format_line(line: str) -> str:
            return split_line(line.split("#")[0])

        def normalize_line(line: str) -> str:
            return line.strip()

        input_lines = [
            format_line(n_line)
            for line in input_file.read_text().splitlines(keepends=False)
            if line and not (n_line := normalize_line(line)).startswith("#")
        ]
        cls_constants = cls(**dict(input_lines))
        cls_constants.correct_values()
        return cls_constants

    def correct_values(self):
        """
        Corrects values that require extra operations, such as theta_max and prop_space.
        """
        self.theta_max *= np.pi
        self.prop_space /= np.sqrt(2.0)
