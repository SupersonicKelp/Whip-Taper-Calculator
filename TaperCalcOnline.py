# =============================================================================
# Created by Joshua Merrell
# =============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st


# =============================================================================
# ASSUMPTIONS / INTERPRETATION
# =============================================================================
# 1) The state generator is construction-driven. It determines the sequence of
#    overlay drops and internal cuts from the build configuration.
# 2) The taper family is then applied to the generated mass-density ladder.
# 3) Families are interpreted as follows:
#       - linear:
#           row-start linear ratio, mass-based
#           profile_i = 1 - i/N
#           raw_delta_i ~ profile_i / mu_i
#
#       - inverse_root_curve:
#           the user's original row-start root-curve rule
#           profile_i = sqrt(1 - i/N)
#           raw_delta_i ~ profile_i / mu_i
#
#       - EXPERIMENTAL: NewRootCurve:
#           continuous exact segment-average version of sqrt(1 - s)
#           raw_delta_i ~ avg_over_segment[sqrt(1 - s)] / mu_i
#
#       - impedance_gradient:
#           matches the user's working analytical program:
#           allocate segment lengths from local drops in ln(mu)
#           using:
#               drops = ln(mu_i) - ln(mu_{i+1})
#               base[0]   = drops[0]
#               base[-1]  = drops[-1]
#               base[i]   = 0.5 * (drops[i-1] + drops[i]) for interior states
#           then normalize base.
#
# 4) Exact taper math is kept separate from the build-helper output.
# 5) Build-helper output rounds cumulative marks to the chosen quantum.
# 6) If Min build move <= 0, the helper does rounding only and does NOT impose
#    a minimum move clamp.
# 7) State-generation shop rules:
#       - belly/core cuts happen before dropped-overlay cuts
#       - avoid back-to-back overlay drops whenever any cut is available
#       - optional double-drop mode allows two overlay strands to drop at once
# 8) Back-to-back drops may still happen at the bare tail if there is literally
#    nothing left to cut under the overlay.
# =============================================================================


# =============================================================================
# MATERIAL PRESETS
# =============================================================================
MATERIAL_PRESETS = {
    "Custom": None,
    "gutted_550 (0.11 g/in)": 0.11,
    "ungutted_550 (0.16 g/in)": 0.16,
    "gutted_275 (0.04 g/in)": 0.04,
    "ungutted_275 (0.10 g/in)": 0.10,
    "shot_loaded (2.20 g/in)": 2.20,
    "Flat Spectra (0.05 g/in)": 0.05,
    "Flat Dacron (0.03 g/in)": 0.03,
    "Ungutted Dacron (0.19 g/in)": 0.19,
    "Gutted Dacron (0.10 g/in)": 0.10,
    "Ungutted DynaX (0.16 g/in)": 0.16,
    "Gutted DynaX (0.12 g/in)": 0.12,
}

TAPER_OPTIONS = [
    "linear",
    "inverse_root_curve",
    "EXPERIMENTAL: NewRootCurve",
    "impedance_gradient",
]

DROP_STYLE_OPTIONS = [
    "single",
    "double",
]

DROP_LOCK_EVENTS = 2
BUILD_QUANTUM_IN = 1.0 / 8.0
BUILD_MIN_DELTA_IN = 7.0 / 8.0

# Overlay-drop spacing targets
DROP_SPACING_ROOT_CUTS = 3
DROP_SPACING_TIP_CUTS = 1
# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class LayerConfig:
    name: str
    count: int
    strand_mass_g_per_in: float
    braided: bool
    enabled: bool = True


@dataclass
class StateRow:
    label: str
    overlay_count: int
    support_count: int
    mu_g_per_in: float


# =============================================================================
# MATERIAL HELPERS
# =============================================================================
def material_mass(selection: str, custom_text: str) -> float:
    if selection not in MATERIAL_PRESETS:
        raise ValueError(f"Unknown material selection: {selection}")

    preset = MATERIAL_PRESETS[selection]
    if preset is not None:
        return float(preset)

    try:
        value = float(custom_text)
    except Exception as exc:
        raise ValueError("Custom mass must be a valid number.") from exc

    if value <= 0.0:
        raise ValueError("Custom mass must be positive.")
    return value


# =============================================================================
# MASS / GEOMETRY MODEL
# =============================================================================
def state_mu(
    overlay_active: int,
    dropped_overlay_underlay: int,
    belly_counts: List[int],
    core_counts: List[int],
    overlay: LayerConfig,
    bellies: List[LayerConfig],
    core_groups: List[LayerConfig],
    braided_factor: float,
) -> float:
    """
    Axial linear density model.

    - Active overlay strands are braided.
    - Dropped overlay strands become straight underlay.
    - Bellies are braided.
    - Core groups are straight unless configured otherwise.
    """
    mu = 0.0

    mu += overlay_active * overlay.strand_mass_g_per_in * braided_factor
    mu += dropped_overlay_underlay * overlay.strand_mass_g_per_in

    for count, belly in zip(belly_counts, bellies):
        if belly.enabled and belly.count > 0:
            factor = braided_factor if belly.braided else 1.0
            mu += count * belly.strand_mass_g_per_in * factor

    for count, core in zip(core_counts, core_groups):
        if core.enabled and core.count > 0:
            factor = braided_factor if core.braided else 1.0
            mu += count * core.strand_mass_g_per_in * factor

    return mu


def support_count_total(
    dropped_overlay_underlay: int,
    belly_counts: List[int],
    core_counts: List[int],
) -> int:
    return dropped_overlay_underlay + sum(belly_counts) + sum(core_counts)


def inner_area_proxy(
    dropped_overlay_underlay: int,
    belly_counts: List[int],
    core_counts: List[int],
    overlay: LayerConfig,
    bellies: List[LayerConfig],
    core_groups: List[LayerConfig],
) -> float:
    """
    Relative inner-bundle thickness proxy.

    Assumption:
    cross-sectional area is proportional to strand mass per strand-inch.
    """
    area = 0.0

    area += dropped_overlay_underlay * overlay.strand_mass_g_per_in

    for count, belly in zip(belly_counts, bellies):
        if belly.enabled and belly.count > 0:
            area += count * belly.strand_mass_g_per_in

    for count, core in zip(core_counts, core_groups):
        if core.enabled and core.count > 0:
            area += count * core.strand_mass_g_per_in

    return area


# =============================================================================
# STATE GENERATOR
# =============================================================================
def first_nonzero_index(values: List[int]) -> Optional[int]:
    for i, v in enumerate(values):
        if v > 0:
            return i
    return None

def required_cuts_before_next_drop(
    overlay_active: int,
    overlay_start: int,
    overlay_end: int,
) -> int:
    """
    Generalized drop-spacing rule.

    Early in the whip:
        require more cuts between drops.
    Late in the whip:
        require fewer cuts between drops.

    This scales automatically with different overlay counts.
    """
    total_drop_strands = overlay_start - overlay_end
    if total_drop_strands <= 0:
        return 1

    progress = (overlay_start - overlay_active) / total_drop_strands
    progress = min(1.0, max(0.0, progress))

    target = DROP_SPACING_TIP_CUTS + (
        DROP_SPACING_ROOT_CUTS - DROP_SPACING_TIP_CUTS
    ) * (1.0 - progress)

    return max(DROP_SPACING_TIP_CUTS, int(round(target)))

def generate_state_rows(
    overlay: LayerConfig,
    bellies: List[LayerConfig],
    core_groups: List[LayerConfig],
    braided_factor: float,
    overlay_end: int,
    drop_style: str,
) -> List[StateRow]:
    """
    Dynamic state generator.

    This is construction-driven, not taper-driven.

    Logic:
    - Overlay drops are interleaved with internal cuts based on a thickness rule.
    - Bellies are cut before core.
    - Core groups are cut in listed order.
    - Dropped overlay strands become underlay immediately after a drop.
    - Dropped overlay strands may be cut only after DROP_LOCK_EVENTS later moves,
      unless forced at the very end.
    - Core/belly cuts happen before dropped-overlay cuts.
    - Avoid back-to-back drops whenever any cut is available.
    - Optional double-drop mode drops two overlay strands at once when allowed.
    - Early in the whip, require more cuts between drops; late in the whip,
      require fewer cuts between drops.
    """
    if overlay.count < overlay_end:
        raise ValueError("Overlay count cannot be smaller than the chosen ending overlay count.")
    if overlay_end not in {2, 4}:
        raise ValueError("Ending overlay count must be 2 or 4.")
    if drop_style not in DROP_STYLE_OPTIONS:
        raise ValueError(f"Drop style must be one of: {', '.join(DROP_STYLE_OPTIONS)}")

    if drop_style == "double" and ((overlay.count - overlay_end) % 2 != 0):
        raise ValueError(
            "Double drop mode requires overlay count and end count to have the same parity "
            "(so the overlay can reach the chosen end using only 2-strand drops)."
        )

    overlay_active = overlay.count
    belly_counts = [b.count if b.enabled else 0 for b in bellies]
    core_counts = [c.count if c.enabled else 0 for c in core_groups]

    dropped_overlay_events = deque()

    def dropped_count() -> int:
        return len(dropped_overlay_events)

    initial_inner_area = inner_area_proxy(
        dropped_overlay_underlay=dropped_count(),
        belly_counts=belly_counts,
        core_counts=core_counts,
        overlay=overlay,
        bellies=bellies,
        core_groups=core_groups,
    )

    if initial_inner_area <= 0.0:
        raise ValueError(
            "Initial inner support is zero. You need some belly and/or core support under the overlay."
        )

    rows: List[StateRow] = []

    rows.append(StateRow(
        label=f"Start ({overlay_active}-{support_count_total(dropped_count(), belly_counts, core_counts)})",
        overlay_count=overlay_active,
        support_count=support_count_total(dropped_count(), belly_counts, core_counts),
        mu_g_per_in=state_mu(
            overlay_active=overlay_active,
            dropped_overlay_underlay=dropped_count(),
            belly_counts=belly_counts,
            core_counts=core_counts,
            overlay=overlay,
            bellies=bellies,
            core_groups=core_groups,
            braided_factor=braided_factor,
        ),
    ))

    last_move_type = "start"
    cuts_since_last_drop = 0

    while True:
        support_before = support_count_total(dropped_count(), belly_counts, core_counts)

        if overlay_active == overlay_end and support_before == 0:
            break

        steps_done = len(rows) - 1

        def any_internal_cut_available_now() -> bool:
            # Actual desired cut order:
            # bellies first, then core, then eligible dropped-overlay cuts.
            if any(v > 0 for v in belly_counts):
                return True
            if any(v > 0 for v in core_counts):
                return True
            return bool(
                dropped_overlay_events and
                (steps_done - dropped_overlay_events[0] >= DROP_LOCK_EVENTS)
            )

        def should_drop_n(n_drop: int) -> bool:
            if overlay_active - n_drop < overlay_end:
                return False

            current_inner_area = inner_area_proxy(
                dropped_overlay_underlay=dropped_count(),
                belly_counts=belly_counts,
                core_counts=core_counts,
                overlay=overlay,
                bellies=bellies,
                core_groups=core_groups,
            )

            if current_inner_area <= 0.0:
                # Bare twist-braid tail
                return True

            a_overlay = overlay.strand_mass_g_per_in
            overlay_new = overlay_active - n_drop
            target_after = initial_inner_area * (overlay_new / overlay.count) ** 2
            threshold_pre = target_after - n_drop * a_overlay

            if overlay_new >= 4:
                threshold_pre = max(n_drop * a_overlay, threshold_pre)

            return current_inner_area <= threshold_pre + 1e-12

        allow_drop_now = not (
            last_move_type == "drop" and any_internal_cut_available_now()
        )

        required_cuts = required_cuts_before_next_drop(
            overlay_active=overlay_active,
            overlay_start=overlay.count,
            overlay_end=overlay_end,
        )

        spacing_satisfied = (
            cuts_since_last_drop >= required_cuts
            or not any_internal_cut_available_now()
        )

        if allow_drop_now and spacing_satisfied and overlay_active > overlay_end:
            n_drop = 0

            if drop_style == "double":
                # Tail exception:
                # if there is no support left and we are at 4-over or below,
                # do single drops because 4 -> 2 is not a valid double-drop transition.
                bare_tail = (support_before == 0)

                if bare_tail and overlay_active <= 4:
                    if should_drop_n(1):
                        n_drop = 1
                else:
                    # Everywhere else in double mode, every drop must be 2 strands.
                    if should_drop_n(2):
                        n_drop = 2
            else:
                if should_drop_n(1):
                    n_drop = 1

            if n_drop > 0:
                old_overlay = overlay_active
                old_support = support_before

                overlay_active -= n_drop

                if support_before > 0:
                    for _ in range(n_drop):
                        dropped_overlay_events.append(steps_done + 1)

                support_after = support_count_total(dropped_count(), belly_counts, core_counts)

                if n_drop == 1:
                    label = f"Drop overlay ({old_overlay}-{old_support} -> {overlay_active}-{support_after})"
                else:
                    label = f"Double drop overlay ({old_overlay}-{old_support} -> {overlay_active}-{support_after})"

                rows.append(StateRow(
                    label=label,
                    overlay_count=overlay_active,
                    support_count=support_after,
                    mu_g_per_in=state_mu(
                        overlay_active=overlay_active,
                        dropped_overlay_underlay=dropped_count(),
                        belly_counts=belly_counts,
                        core_counts=core_counts,
                        overlay=overlay,
                        bellies=bellies,
                        core_groups=core_groups,
                        braided_factor=braided_factor,
                    ),
                ))
                last_move_type = "drop"
                cuts_since_last_drop = 0
                continue
        # ---------------------------------------------------------
        # Internal cuts:
        # Bellies first, core second, dropped overlay last.
        # ---------------------------------------------------------
        belly_idx = first_nonzero_index(belly_counts)
        if belly_idx is not None:
            old_overlay = overlay_active
            old_support = support_before

            belly_counts[belly_idx] -= 1
            support_after = support_count_total(dropped_count(), belly_counts, core_counts)

            rows.append(StateRow(
                label=f"Cut belly {belly_idx + 1} ({old_overlay}-{old_support} -> {old_overlay}-{support_after})",
                overlay_count=old_overlay,
                support_count=support_after,
                mu_g_per_in=state_mu(
                    overlay_active=overlay_active,
                    dropped_overlay_underlay=dropped_count(),
                    belly_counts=belly_counts,
                    core_counts=core_counts,
                    overlay=overlay,
                    bellies=bellies,
                    core_groups=core_groups,
                    braided_factor=braided_factor,
                ),
            ))
            last_move_type = "cut"
            cuts_since_last_drop += 1
            continue

        core_idx = first_nonzero_index(core_counts)
        if core_idx is not None:
            old_overlay = overlay_active
            old_support = support_before

            core_counts[core_idx] -= 1
            support_after = support_count_total(dropped_count(), belly_counts, core_counts)

            rows.append(StateRow(
                label=f"Cut core {core_idx + 1} ({old_overlay}-{old_support} -> {old_overlay}-{support_after})",
                overlay_count=old_overlay,
                support_count=support_after,
                mu_g_per_in=state_mu(
                    overlay_active=overlay_active,
                    dropped_overlay_underlay=dropped_count(),
                    belly_counts=belly_counts,
                    core_counts=core_counts,
                    overlay=overlay,
                    bellies=bellies,
                    core_groups=core_groups,
                    braided_factor=braided_factor,
                ),
            ))
            last_move_type = "cut"
            cuts_since_last_drop += 1
            continue

        if dropped_overlay_events and (steps_done - dropped_overlay_events[0] >= DROP_LOCK_EVENTS):
            old_overlay = overlay_active
            old_support = support_before

            dropped_overlay_events.popleft()
            support_after = support_count_total(dropped_count(), belly_counts, core_counts)

            rows.append(StateRow(
                label=f"Cut dropped overlay ({old_overlay}-{old_support} -> {overlay_active}-{support_after})",
                overlay_count=overlay_active,
                support_count=support_after,
                mu_g_per_in=state_mu(
                    overlay_active=overlay_active,
                    dropped_overlay_underlay=dropped_count(),
                    belly_counts=belly_counts,
                    core_counts=core_counts,
                    overlay=overlay,
                    bellies=bellies,
                    core_groups=core_groups,
                    braided_factor=braided_factor,
                ),
            ))
            last_move_type = "cut"
            cuts_since_last_drop += 1
            continue

        # Forced cleanup at the very end if nothing else remains
        if dropped_overlay_events:
            old_overlay = overlay_active
            old_support = support_before

            dropped_overlay_events.popleft()
            support_after = support_count_total(dropped_count(), belly_counts, core_counts)

            rows.append(StateRow(
                label=f"Cut dropped overlay ({old_overlay}-{old_support} -> {overlay_active}-{support_after})",
                overlay_count=overlay_active,
                support_count=support_after,
                mu_g_per_in=state_mu(
                    overlay_active=overlay_active,
                    dropped_overlay_underlay=dropped_count(),
                    belly_counts=belly_counts,
                    core_counts=core_counts,
                    overlay=overlay,
                    bellies=bellies,
                    core_groups=core_groups,
                    braided_factor=braided_factor,
                ),
            ))
            last_move_type = "cut"
            cuts_since_last_drop += 1
            continue

        raise ValueError("Planner got stuck with no valid move.")

    mu_values = np.array([r.mu_g_per_in for r in rows], dtype=float)
    if np.any(np.diff(mu_values) >= 0.0):
        raise ValueError(
            "Generated mass-density ladder is not strictly decreasing. Check your configuration."
        )

    return rows

# =============================================================================
# TAPER ENGINES
# =============================================================================
def equal_log_drop_fractions_from_mu(mu_values: np.ndarray) -> np.ndarray:
    """
    Match the user's working analytical impedance-gradient code exactly.

    Allocate segment weights from local drops in ln(mu).
    """
    mu_values = np.asarray(mu_values, dtype=float)

    if mu_values.ndim != 1 or len(mu_values) == 0:
        return np.array([], dtype=float)

    if np.any(mu_values <= 0.0):
        raise ValueError("All mass-density values must be positive.")
    if np.any(np.diff(mu_values) >= 0.0):
        raise ValueError("Mass-density ladder must be strictly decreasing.")

    u = np.log(mu_values)
    drops = u[:-1] - u[1:]

    if np.any(drops <= 0.0):
        raise ValueError("Density ladder must decrease strictly.")

    base = np.empty_like(mu_values)
    base[0] = drops[0]
    base[-1] = drops[-1]

    if len(mu_values) > 2:
        base[1:-1] = 0.5 * (drops[:-1] + drops[1:])

    base = np.clip(base, 1e-15, None)
    return base / np.sum(base)


def exact_normalized_deltas_from_family(
    mu_values: np.ndarray,
    family: str,
) -> Tuple[np.ndarray, np.ndarray]:
    mu_values = np.asarray(mu_values, dtype=float)

    if mu_values.ndim != 1 or len(mu_values) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    if np.any(mu_values <= 0.0):
        raise ValueError("All mass-density values must be positive.")
    if np.any(np.diff(mu_values) >= 0.0):
        raise ValueError("Mass-density ladder must be strictly decreasing.")

    n_rows = len(mu_values)
    s = np.arange(n_rows, dtype=float) / n_rows

    if family == "linear":
        profile = 1.0 - s
        raw_delta = profile / mu_values

    elif family == "inverse_root_curve":
        profile = np.sqrt(1.0 - s)
        raw_delta = profile / mu_values

    elif family == "EXPERIMENTAL: NewRootCurve":
        i = np.arange(n_rows, dtype=float)
        a = i / n_rows
        b = (i + 1.0) / n_rows

        profile_avg = (2.0 * n_rows / 3.0) * (
            np.power(1.0 - a, 1.5) - np.power(1.0 - b, 1.5)
        )
        raw_delta = profile_avg / mu_values

    elif family == "impedance_gradient":
        # Match spreadsheet logic:
        # FJ = impedance fractions from ln(mu)
        # FK = cumulative row-start impedance profile
        # EH = FK / EN
        # final delta = normalized(EH)
        imp_frac = equal_log_drop_fractions_from_mu(mu_values)
        imp_node_start = np.concatenate(([1.0], 1.0 - np.cumsum(imp_frac[:-1])))
        raw_delta = imp_node_start / mu_values

    else:
        raise ValueError(f"Unknown taper family: {family}")

    if np.any(raw_delta <= 0.0):
        raise ValueError("Computed raw segment lengths must be positive.")

    norm_delta = raw_delta / np.sum(raw_delta)
    node_end = 1.0 - np.cumsum(norm_delta)
    node_end[-1] = 0.0

    return node_end, norm_delta


# =============================================================================
# BUILDABLE ROUNDING HELPERS
# =============================================================================
def buildable_nodes_from_exact(
    exact_inch_nodes: np.ndarray,
    total_length_in: float,
    quantum_in: float = BUILD_QUANTUM_IN,
    min_delta_in: float = BUILD_MIN_DELTA_IN,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create shop-friendly node locations from the exact nodes.

    If min_delta_in <= 0:
        round only, no minimum clamp
    Otherwise:
        round and enforce a practical minimum move
    """
    exact_inch_nodes = np.asarray(exact_inch_nodes, dtype=float)

    if len(exact_inch_nodes) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    if total_length_in <= 0.0:
        raise ValueError("Total length must be positive.")
    if quantum_in <= 0.0:
        raise ValueError("Build quantum must be positive.")

    total_units = int(round(total_length_in / quantum_in))
    target_units = np.round(exact_inch_nodes / quantum_in).astype(int)
    n = len(target_units)

    if min_delta_in <= 0.0:
        u = target_units.copy()

        for i in range(1, n):
            if u[i] < u[i - 1]:
                u[i] = u[i - 1]

        u[-1] = total_units

        for i in range(n - 2, -1, -1):
            if u[i] > u[i + 1]:
                u[i] = u[i + 1]

        u[-1] = total_units

        build_nodes = u.astype(float) * quantum_in
        build_deltas = np.diff(np.concatenate(([0.0], build_nodes)))
        return build_nodes, build_deltas

    min_units = max(1, int(round(min_delta_in / quantum_in)))
    max_feasible_min_units = max(1, total_units // n)
    min_units = min(min_units, max_feasible_min_units)

    lower = min_units * (np.arange(n) + 1)
    upper = total_units - min_units * (np.arange(n - 1, -1, -1))

    u = np.clip(target_units, lower, upper)

    for i in range(1, n):
        if u[i] < u[i - 1] + min_units:
            u[i] = u[i - 1] + min_units

    u[-1] = total_units

    for i in range(n - 2, -1, -1):
        max_allowed = u[i + 1] - min_units
        if u[i] > max_allowed:
            u[i] = max_allowed

    for i in range(1, n):
        if u[i] < u[i - 1] + min_units:
            u[i] = u[i - 1] + min_units

    u[-1] = total_units

    build_nodes = u.astype(float) * quantum_in
    build_deltas = np.diff(np.concatenate(([0.0], build_nodes)))

    return build_nodes, build_deltas


def format_inches_as_eighths(value: float) -> str:
    eighths = int(round(value * 8.0))
    whole = eighths // 8
    frac = eighths % 8

    if frac == 0:
        return str(whole)

    g = math.gcd(frac, 8)
    num = frac // g
    den = 8 // g

    if whole == 0:
        return f"{num}/{den}"
    return f"{whole} {num}/{den}"


# =============================================================================
# TABLE BUILDER
# =============================================================================
def build_table_rows(
    thong_length_in: float,
    overlay: LayerConfig,
    bellies: List[LayerConfig],
    core_groups: List[LayerConfig],
    braided_factor: float,
    taper_family: str,
    overlay_end: int,
    drop_style: str,
    build_quantum_in: float,
    build_min_delta_in: float,
) -> List[dict]:
    state_rows = generate_state_rows(
        overlay=overlay,
        bellies=bellies,
        core_groups=core_groups,
        braided_factor=braided_factor,
        overlay_end=overlay_end,
        drop_style=drop_style,
    )

    mu_values = np.array([r.mu_g_per_in for r in state_rows], dtype=float)

    node_end_exact, norm_delta_exact = exact_normalized_deltas_from_family(
        mu_values,
        taper_family,
    )

    inch_delta_exact = norm_delta_exact * thong_length_in
    inch_node_exact = np.cumsum(inch_delta_exact)

    inch_node_build, inch_delta_build = buildable_nodes_from_exact(
        exact_inch_nodes=inch_node_exact,
        total_length_in=thong_length_in,
        quantum_in=build_quantum_in,
        min_delta_in=build_min_delta_in,
    )

    rows = []
    for i, sr in enumerate(state_rows):
        rows.append({
            "row_index": i + 1,
            "move": sr.label,
            "normalized_node": node_end_exact[i],
            "normalized_delta": norm_delta_exact[i],
            "inch_node": inch_node_exact[i],
            "inch_delta": inch_delta_exact[i],
            "build_node": inch_node_build[i],
            "build_delta": inch_delta_build[i],
            "build_node_text": format_inches_as_eighths(inch_node_build[i]),
            "build_delta_text": format_inches_as_eighths(inch_delta_build[i]),
            "mu_g_per_in": sr.mu_g_per_in,
        })

    return rows

# =============================================================================
# STREAMLIT UI
# =============================================================================
def beginner_preset() -> dict:
    return {
        "thong_length_in": 72.0,
        "family": "impedance_gradient",
        "braided_factor": float(math.sqrt(2.0)),
        "end_value": 2,
        "build_quantum_in": float(BUILD_QUANTUM_IN),
        "build_min_delta_in": float(BUILD_MIN_DELTA_IN),
        "drop_style": "single",

        "overlay_count": 12,
        "overlay_material": "gutted_550 (0.11 g/in)",
        "overlay_custom": 0.11,

        "belly1_enabled": False,
        "belly1_count": 0,
        "belly1_material": "gutted_550 (0.11 g/in)",
        "belly1_custom": 0.11,

        "belly2_enabled": False,
        "belly2_count": 0,
        "belly2_material": "gutted_550 (0.11 g/in)",
        "belly2_custom": 0.11,

        "belly3_enabled": False,
        "belly3_count": 0,
        "belly3_material": "gutted_550 (0.11 g/in)",
        "belly3_custom": 0.11,

        "core1_enabled": True,
        "core1_count": 12,
        "core1_material": "gutted_550 (0.11 g/in)",
        "core1_custom": 0.11,

        "core2_enabled": False,
        "core2_count": 0,
        "core2_material": "gutted_550 (0.11 g/in)",
        "core2_custom": 0.11,

        "core3_enabled": False,
        "core3_count": 0,
        "core3_material": "gutted_550 (0.11 g/in)",
        "core3_custom": 0.11,
    }


def init_session_state() -> None:
    defaults = beginner_preset()
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def load_beginner_preset() -> None:
    st.session_state.update(beginner_preset())


def layer_editor(
    prefix: str,
    title: str,
    show_enable: bool = True,
) -> None:
    st.markdown(f"### {title}")
    cols = st.columns([1.1, 1.0, 2.2, 1.4])

    if show_enable:
        cols[0].checkbox("Enabled", key=f"{prefix}_enabled")
    else:
        cols[0].markdown("**Enabled:** yes")

    cols[1].number_input(
        "Count",
        min_value=0,
        step=1,
        key=f"{prefix}_count",
    )

    cols[2].selectbox(
        "Material",
        options=list(MATERIAL_PRESETS.keys()),
        key=f"{prefix}_material",
    )

    selected = st.session_state[f"{prefix}_material"]
    preset = MATERIAL_PRESETS[selected]

    if preset is None:
        cols[3].number_input(
            "Custom mass (g/in)",
            min_value=0.000001,
            step=0.01,
            format="%.5f",
            key=f"{prefix}_custom",
        )
    else:
        st.session_state[f"{prefix}_custom"] = float(preset)
        cols[3].number_input(
            "Mass (g/in)",
            min_value=0.0,
            step=0.01,
            format="%.5f",
            value=float(preset),
            disabled=True,
            key=f"{prefix}_preset_mass_display",
        )


def layer_config_from_state(
    prefix: str,
    name: str,
    braided: bool,
    force_enabled: bool = False,
) -> LayerConfig:
    enabled = True if force_enabled else bool(st.session_state.get(f"{prefix}_enabled", True))

    count = int(st.session_state[f"{prefix}_count"])
    if not enabled:
        count = 0

    mass = material_mass(
        st.session_state[f"{prefix}_material"],
        str(st.session_state.get(f"{prefix}_custom", 0.0)),
    )

    return LayerConfig(
        name=name,
        count=count,
        strand_mass_g_per_in=mass,
        braided=braided,
        enabled=enabled,
    )


def make_display_dataframe(rows: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).copy()

    for col in ["normalized_node", "normalized_delta"]:
        df[col] = df[col].map(lambda x: f"{x:.9f}")

    for col in ["inch_node", "inch_delta", "build_node", "build_delta", "mu_g_per_in"]:
        df[col] = df[col].map(lambda x: f"{x:.6f}")

    df = df.rename(columns={
        "row_index": "Row #",
        "move": "Move / State",
        "normalized_node": "Normalized Node",
        "normalized_delta": "Normalized Delta",
        "inch_node": "Exact Inch Node",
        "inch_delta": "Exact Delta Inch",
        "build_node": "Build Node",
        "build_delta": "Build Delta",
        "build_node_text": 'Build Node (1/8")',
        "build_delta_text": 'Build Delta (1/8")',
        "mu_g_per_in": "mu (g/in)",
    })

    return df


def main() -> None:
    st.set_page_config(page_title="Dynamic Whip Taper Planner", layout="wide")
    init_session_state()

    st.title("Dynamic Whip Taper Planner")
    st.caption(
        "Same math/state engine, Streamlit front end. "
        "Single-link deploy friendly."
    )

    top_cols = st.columns([1, 4])
    if top_cols[0].button("Load Beginner Preset"):
        load_beginner_preset()
        st.rerun()

    with st.sidebar:
        st.header("General Inputs")

        st.number_input(
            "Thong length (in)",
            min_value=0.001,
            step=1.0,
            format="%.3f",
            key="thong_length_in",
        )

        st.selectbox(
            "Taper family",
            options=TAPER_OPTIONS,
            key="family",
        )

        st.number_input(
            "Braided factor",
            min_value=1.000001,
            step=0.01,
            format="%.9f",
            key="braided_factor",
        )

        st.selectbox(
            "End at",
            options=[2, 4],
            key="end_value",
        )

        st.number_input(
            "Build quantum (in)",
            min_value=0.000001,
            step=0.125,
            format="%.3f",
            key="build_quantum_in",
        )

        st.number_input(
            "Min build move (in)",
            step=0.125,
            format="%.3f",
            key="build_min_delta_in",
        )

        st.selectbox(
            "Drop style",
            options=DROP_STYLE_OPTIONS,
            key="drop_style",
        )

        st.markdown("---")
        st.caption("In double-drop mode, the bare 4-over tail may still single-drop to 3 and 2.")

    tab_overlay, tab_bellies, tab_core = st.tabs(["Overlay", "Bellies", "Core Groups"])

    with tab_overlay:
        layer_editor("overlay", "Overlay", show_enable=False)

    with tab_bellies:
        layer_editor("belly1", "Belly 1", show_enable=True)
        layer_editor("belly2", "Belly 2", show_enable=True)
        layer_editor("belly3", "Belly 3", show_enable=True)

    with tab_core:
        layer_editor("core1", "Core 1", show_enable=True)
        layer_editor("core2", "Core 2", show_enable=True)
        layer_editor("core3", "Core 3", show_enable=True)
        st.info("Put shot-loaded core groups earlier. Put the final continuation core group last.")

    try:
        thong_length_in = float(st.session_state["thong_length_in"])
        family = str(st.session_state["family"])
        braided_factor = float(st.session_state["braided_factor"])
        overlay_end = int(st.session_state["end_value"])
        build_quantum_in = float(st.session_state["build_quantum_in"])
        build_min_delta_in = float(st.session_state["build_min_delta_in"])
        drop_style = str(st.session_state["drop_style"])

        overlay = layer_config_from_state("overlay", "overlay", braided=True, force_enabled=True)
        bellies = [
            layer_config_from_state("belly1", "belly1", braided=True),
            layer_config_from_state("belly2", "belly2", braided=True),
            layer_config_from_state("belly3", "belly3", braided=True),
        ]
        core_groups = [
            layer_config_from_state("core1", "core1", braided=False),
            layer_config_from_state("core2", "core2", braided=False),
            layer_config_from_state("core3", "core3", braided=False),
        ]

        rows = build_table_rows(
            thong_length_in=thong_length_in,
            overlay=overlay,
            bellies=bellies,
            core_groups=core_groups,
            braided_factor=braided_factor,
            taper_family=family,
            overlay_end=overlay_end,
            drop_style=drop_style,
            build_quantum_in=build_quantum_in,
            build_min_delta_in=build_min_delta_in,
        )

    except Exception as exc:
        st.error(str(exc))
        st.stop()

    norm_sum = sum(r["normalized_delta"] for r in rows)
    inch_sum = sum(r["inch_delta"] for r in rows)
    build_sum = sum(r["build_delta"] for r in rows)
    min_exact = min(r["inch_delta"] for r in rows)
    min_build = min(r["build_delta"] for r in rows)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{len(rows)}")
    m2.metric("Taper", family)
    m3.metric("Drop style", drop_style)
    m4.metric("End at", f"{overlay_end}-0")

    st.markdown(
        f"""
**Thong length:** {thong_length_in:.3f} in  
**Normalized delta sum:** {norm_sum:.9f}  
**Exact delta inch sum:** {inch_sum:.6f}  
**Build delta inch sum:** {build_sum:.6f}  
**Build quantum:** {build_quantum_in:.3f} in  
**Requested min build move:** {build_min_delta_in:.3f} in  
**Smallest exact move:** {min_exact:.6f} in  
**Smallest build move:** {min_build:.6f} in
"""
    )

    st.subheader("Taper Table")
    display_df = make_display_dataframe(rows)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    raw_df = pd.DataFrame(rows)
    csv_bytes = raw_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name="dynamic_whip_taper_plan.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()