from scipy.interpolate import PchipInterpolator
import numpy as np


def create_phase(swing_duration, stance_duration, strict_relaxer, stance_mode, FREQ=40):
    # NOTE: these times are being converted from time in seconds to phaselength
    R_swing = np.array([0.0, swing_duration]) * FREQ
    first_dblstance = (
        np.array([swing_duration, swing_duration + stance_duration]) * FREQ
    )
    L_swing = (
        np.array(
            [swing_duration + stance_duration, 2 * swing_duration + stance_duration]
        )
        * FREQ
    )
    second_dblstance = (
        np.array(
            [
                2 * swing_duration + stance_duration,
                2 * (swing_duration + stance_duration),
            ]
        )
        * FREQ
    )

    r_frc_phase_points = np.zeros((2, 8))
    r_vel_phase_points = np.zeros((2, 8))
    l_frc_phase_points = np.zeros((2, 8))
    l_vel_phase_points = np.zeros((2, 8))

    R_swing_relax_offset = (R_swing[1] - R_swing[0]) * strict_relaxer
    l_frc_phase_points[0, 0] = r_frc_phase_points[0, 0] = (
        R_swing[0] + R_swing_relax_offset
    )
    l_frc_phase_points[0, 1] = r_frc_phase_points[0, 1] = (
        R_swing[1] - R_swing_relax_offset
    )
    l_vel_phase_points[0, 0] = r_vel_phase_points[0, 0] = (
        R_swing[0] + R_swing_relax_offset
    )
    l_vel_phase_points[0, 1] = r_vel_phase_points[0, 1] = (
        R_swing[1] - R_swing_relax_offset
    )
    # During R swing we want foot velocities and don't want foot forces
    l_vel_phase_points[1, :2] = r_frc_phase_points[1, :2] = np.negative(
        np.ones(2)
    )  # penalize l vel and r force
    l_frc_phase_points[1, :2] = r_vel_phase_points[1, :2] = np.ones(
        2
    )  # incentivize l force and r vel

    dbl_stance_relax_offset = (first_dblstance[1] - first_dblstance[0]) * strict_relaxer
    l_frc_phase_points[0, 2] = r_frc_phase_points[0, 2] = (
        first_dblstance[0] + dbl_stance_relax_offset
    )
    l_frc_phase_points[0, 3] = r_frc_phase_points[0, 3] = (
        first_dblstance[1] - dbl_stance_relax_offset
    )
    l_vel_phase_points[0, 2] = r_vel_phase_points[0, 2] = (
        first_dblstance[0] + dbl_stance_relax_offset
    )
    l_vel_phase_points[0, 3] = r_vel_phase_points[0, 3] = (
        first_dblstance[1] - dbl_stance_relax_offset
    )
    if stance_mode == "aerial":
        # During aerial we want foot velocities and don't want foot forces
        # During grounded walking we want foot forces and don't want velocities
        l_frc_phase_points[1, 2:4] = r_frc_phase_points[1, 2:4] = np.negative(
            np.ones(2)
        )  # penalize l and r foot force
        l_vel_phase_points[1, 2:4] = r_vel_phase_points[1, 2:4] = np.ones(
            2
        )  # incentivize l and r foot velocity
    elif stance_mode == "zero":
        l_frc_phase_points[1, 2:4] = r_frc_phase_points[1, 2:4] = np.zeros(2)
        l_vel_phase_points[1, 2:4] = r_vel_phase_points[1, 2:4] = np.zeros(2)
    else:
        # During grounded walking we want foot forces and don't want velocities
        l_frc_phase_points[1, 2:4] = r_frc_phase_points[1, 2:4] = np.ones(
            2
        )  # incentivize l and r foot force
        l_vel_phase_points[1, 2:4] = r_vel_phase_points[1, 2:4] = np.negative(
            np.ones(2)
        )  # penalize l and r foot velocity

    L_swing_relax_offset = (L_swing[1] - L_swing[0]) * strict_relaxer
    l_frc_phase_points[0, 4] = r_frc_phase_points[0, 4] = (
        L_swing[0] + L_swing_relax_offset
    )
    l_frc_phase_points[0, 5] = r_frc_phase_points[0, 5] = (
        L_swing[1] - L_swing_relax_offset
    )
    l_vel_phase_points[0, 4] = r_vel_phase_points[0, 4] = (
        L_swing[0] + L_swing_relax_offset
    )
    l_vel_phase_points[0, 5] = r_vel_phase_points[0, 5] = (
        L_swing[1] - L_swing_relax_offset
    )
    # During L swing we want foot forces and don't want foot velocities (from perspective of R foot)
    l_vel_phase_points[1, 4:6] = r_frc_phase_points[1, 4:6] = np.ones(
        2
    )  # incentivize l vel and r force
    l_frc_phase_points[1, 4:6] = r_vel_phase_points[1, 4:6] = np.negative(
        np.ones(2)
    )  # penalize l force and r vel

    dbl_stance_relax_offset = (
        second_dblstance[1] - second_dblstance[0]
    ) * strict_relaxer
    l_frc_phase_points[0, 6] = r_frc_phase_points[0, 6] = (
        second_dblstance[0] + dbl_stance_relax_offset
    )
    l_frc_phase_points[0, 7] = r_frc_phase_points[0, 7] = (
        second_dblstance[1] - dbl_stance_relax_offset
    )
    l_vel_phase_points[0, 6] = r_vel_phase_points[0, 6] = (
        second_dblstance[0] + dbl_stance_relax_offset
    )
    l_vel_phase_points[0, 7] = r_vel_phase_points[0, 7] = (
        second_dblstance[1] - dbl_stance_relax_offset
    )
    if stance_mode == "aerial":
        # During aerial we want foot velocities and don't want foot forces
        # During grounded walking we want foot forces and don't want velocities
        l_frc_phase_points[1, 6:] = r_frc_phase_points[1, 6:] = np.negative(
            np.ones(2)
        )  # penalize l and r foot force
        l_vel_phase_points[1, 6:] = r_vel_phase_points[1, 6:] = np.ones(
            2
        )  # incentivize l and r foot velocity
    elif stance_mode == "zero":
        l_frc_phase_points[1, 6:] = r_frc_phase_points[1, 6:] = np.zeros(2)
        l_vel_phase_points[1, 6:] = r_vel_phase_points[1, 6:] = np.zeros(2)
    else:
        # During grounded walking we want foot forces and don't want velocities
        l_frc_phase_points[1, 6:] = r_frc_phase_points[1, 6:] = np.ones(
            2
        )  # incentivize l and r foot force
        l_vel_phase_points[1, 6:] = r_vel_phase_points[1, 6:] = np.negative(
            np.ones(2)
        )  # penalize l and r foot velocity

    ## extend the data to three cycles : one before and one after : this ensures continuity

    r_frc_prev_cycle = np.copy(r_frc_phase_points)
    r_vel_prev_cycle = np.copy(r_vel_phase_points)
    l_frc_prev_cycle = np.copy(l_frc_phase_points)
    l_vel_prev_cycle = np.copy(l_vel_phase_points)
    l_frc_prev_cycle[0] = r_frc_prev_cycle[0] = (
        r_frc_phase_points[0] - r_frc_phase_points[0, -1] - dbl_stance_relax_offset
    )
    l_vel_prev_cycle[0] = r_vel_prev_cycle[0] = (
        r_vel_phase_points[0] - r_vel_phase_points[0, -1] - dbl_stance_relax_offset
    )

    r_frc_second_cycle = np.copy(r_frc_phase_points)
    r_vel_second_cycle = np.copy(r_vel_phase_points)
    l_frc_second_cycle = np.copy(l_frc_phase_points)
    l_vel_second_cycle = np.copy(l_vel_phase_points)
    l_frc_second_cycle[0] = r_frc_second_cycle[0] = (
        r_frc_phase_points[0] + r_frc_phase_points[0, -1] + dbl_stance_relax_offset
    )
    l_vel_second_cycle[0] = r_vel_second_cycle[0] = (
        r_vel_phase_points[0] + r_vel_phase_points[0, -1] + dbl_stance_relax_offset
    )

    r_frc_phase_points_repeated = np.hstack(
        (r_frc_prev_cycle, r_frc_phase_points, r_frc_second_cycle)
    )
    r_vel_phase_points_repeated = np.hstack(
        (r_vel_prev_cycle, r_vel_phase_points, r_vel_second_cycle)
    )
    l_frc_phase_points_repeated = np.hstack(
        (l_frc_prev_cycle, l_frc_phase_points, l_frc_second_cycle)
    )
    l_vel_phase_points_repeated = np.hstack(
        (l_vel_prev_cycle, l_vel_phase_points, l_vel_second_cycle)
    )

    ## Create the smoothing function with cubic spline and cutoff at limits -1 and 1
    r_frc_phase_spline = PchipInterpolator(
        r_frc_phase_points_repeated[0], r_frc_phase_points_repeated[1]
    )
    r_vel_phase_spline = PchipInterpolator(
        r_vel_phase_points_repeated[0], r_vel_phase_points_repeated[1]
    )
    l_frc_phase_spline = PchipInterpolator(
        l_frc_phase_points_repeated[0], l_frc_phase_points_repeated[1]
    )
    l_vel_phase_spline = PchipInterpolator(
        l_vel_phase_points_repeated[0], l_vel_phase_points_repeated[1]
    )
    R_clock = [r_frc_phase_spline, r_vel_phase_spline]
    L_clock = [
        l_frc_phase_spline,
        l_vel_phase_spline,
    ]
    return R_clock, L_clock
