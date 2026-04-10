from __future__ import annotations

import math
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import deque

import numpy as np


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
# GUI HELPERS
# =============================================================================
class LayerRow:
    def __init__(
        self,
        parent: ttk.Frame,
        title: str,
        row: int,
        default_enabled: bool,
        default_count: int,
        default_material: str,
        show_enable: bool = True,
    ) -> None:
        self.show_enable = show_enable

        if show_enable:
            self.enabled_var = tk.BooleanVar(value=default_enabled)
            self.enable_widget = ttk.Checkbutton(parent, text=title, variable=self.enabled_var)
            self.enable_widget.grid(row=row, column=0, sticky="w", padx=4, pady=2)
        else:
            self.enabled_var = tk.BooleanVar(value=True)
            ttk.Label(parent, text=title).grid(row=row, column=0, sticky="w", padx=4, pady=2)

        self.count_var = tk.StringVar(value=str(default_count))
        ttk.Entry(parent, textvariable=self.count_var, width=8).grid(row=row, column=1, sticky="w", padx=4)

        self.material_var = tk.StringVar(value=default_material)
        self.material_combo = ttk.Combobox(
            parent,
            textvariable=self.material_var,
            values=list(MATERIAL_PRESETS.keys()),
            state="readonly",
            width=24,
        )
        self.material_combo.grid(row=row, column=2, sticky="w", padx=4)
        self.material_combo.bind("<<ComboboxSelected>>", self._on_material_change)

        self.custom_var = tk.StringVar(value="")
        self.custom_entry = ttk.Entry(parent, textvariable=self.custom_var, width=10)
        self.custom_entry.grid(row=row, column=3, sticky="w", padx=4)

        self._on_material_change()

    def _on_material_change(self, event=None) -> None:
        selection = self.material_var.get()
        preset = MATERIAL_PRESETS.get(selection, None)
        if preset is None:
            self.custom_entry.configure(state="normal")
        else:
            self.custom_var.set(f"{preset:.5f}")
            self.custom_entry.configure(state="disabled")

    def set_values(self, enabled: bool, count: int, material: str, custom: str = "") -> None:
        self.enabled_var.set(enabled)
        self.count_var.set(str(count))
        self.material_var.set(material)
        self.custom_var.set(custom)
        self._on_material_change()

    def to_config(self, name: str, braided: bool, force_enabled: bool = False) -> LayerConfig:
        enabled = True if force_enabled else self.enabled_var.get()

        try:
            count = int(self.count_var.get().strip())
        except Exception as exc:
            raise ValueError(f"{name}: count must be an integer.") from exc

        if count < 0:
            raise ValueError(f"{name}: count cannot be negative.")

        if not enabled:
            count = 0

        mass = material_mass(self.material_var.get(), self.custom_var.get().strip())

        return LayerConfig(
            name=name,
            count=count,
            strand_mass_g_per_in=mass,
            braided=braided,
            enabled=enabled,
        )


# =============================================================================
# GUI APP
# =============================================================================
class DynamicPlannerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Dynamic Whip Taper Planner")
        self.geometry("1600x900")

        self.rows: List[dict] = []

        self._build_ui()
        self.load_beginner_preset()
        self.calculate()

    def _build_ui(self) -> None:
        main = ttk.Frame(self, padding=10)
        main.pack(fill="both", expand=True)

        general = ttk.LabelFrame(main, text="General Inputs", padding=8)
        general.pack(fill="x", pady=4)

        ttk.Label(general, text="Thong length (in)").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self.thong_length_var = tk.StringVar(value="72")
        ttk.Entry(general, textvariable=self.thong_length_var, width=10).grid(row=0, column=1, sticky="w", padx=4)

        ttk.Label(general, text="Taper family").grid(row=0, column=2, sticky="w", padx=4, pady=2)
        self.family_var = tk.StringVar(value="impedance_gradient")
        ttk.Combobox(
            general,
            textvariable=self.family_var,
            values=TAPER_OPTIONS,
            state="readonly",
            width=28,
        ).grid(row=0, column=3, sticky="w", padx=4)

        ttk.Label(general, text="Braided factor").grid(row=0, column=4, sticky="w", padx=4, pady=2)
        self.braided_factor_var = tk.StringVar(value=f"{math.sqrt(2.0):.9f}")
        ttk.Entry(general, textvariable=self.braided_factor_var, width=12).grid(row=0, column=5, sticky="w", padx=4)

        ttk.Label(general, text="End at").grid(row=0, column=6, sticky="w", padx=4, pady=2)
        self.end_var = tk.StringVar(value="2")
        ttk.Combobox(
            general,
            textvariable=self.end_var,
            values=["2", "4"],
            state="readonly",
            width=8,
        ).grid(row=0, column=7, sticky="w", padx=4)

        ttk.Label(general, text="Build quantum (in)").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        self.build_quantum_var = tk.StringVar(value=f"{BUILD_QUANTUM_IN:.3f}")
        ttk.Entry(general, textvariable=self.build_quantum_var, width=10).grid(row=1, column=1, sticky="w", padx=4)

        ttk.Label(general, text="Min build move (in)").grid(row=1, column=2, sticky="w", padx=4, pady=2)
        self.build_min_delta_var = tk.StringVar(value=f"{BUILD_MIN_DELTA_IN:.3f}")
        ttk.Entry(general, textvariable=self.build_min_delta_var, width=10).grid(row=1, column=3, sticky="w", padx=4)

        ttk.Label(general, text="Drop style").grid(row=1, column=4, sticky="w", padx=4, pady=2)
        self.drop_style_var = tk.StringVar(value="single")
        ttk.Combobox(
            general,
            textvariable=self.drop_style_var,
            values=DROP_STYLE_OPTIONS,
            state="readonly",
            width=12,
        ).grid(row=1, column=5, sticky="w", padx=4)

        ttk.Button(general, text="Load Beginner Preset", command=self.load_beginner_preset).grid(
            row=0, column=8, sticky="w", padx=6
        )
        ttk.Button(general, text="Calculate", command=self.calculate).grid(
            row=0, column=9, sticky="w", padx=6
        )
        ttk.Button(general, text="Export CSV", command=self.export_csv).grid(
            row=0, column=10, sticky="w", padx=6
        )

        overlay_frame = ttk.LabelFrame(main, text="Overlay", padding=8)
        overlay_frame.pack(fill="x", pady=4)

        ttk.Label(overlay_frame, text="Layer").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Label(overlay_frame, text="Count").grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(overlay_frame, text="Material").grid(row=0, column=2, sticky="w", padx=4)
        ttk.Label(overlay_frame, text="Custom mass").grid(row=0, column=3, sticky="w", padx=4)

        self.overlay_row = LayerRow(
            overlay_frame,
            title="overlay",
            row=1,
            default_enabled=True,
            default_count=12,
            default_material="gutted_550 (0.11 g/in)",
            show_enable=False,
        )

        belly_frame = ttk.LabelFrame(main, text="Bellies (braided, cuts only)", padding=8)
        belly_frame.pack(fill="x", pady=4)

        ttk.Label(belly_frame, text="Enabled").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Label(belly_frame, text="Count").grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(belly_frame, text="Material").grid(row=0, column=2, sticky="w", padx=4)
        ttk.Label(belly_frame, text="Custom mass").grid(row=0, column=3, sticky="w", padx=4)

        self.belly_rows = [
            LayerRow(belly_frame, "belly1", 1, False, 0, "gutted_550 (0.11 g/in)", show_enable=True),
            LayerRow(belly_frame, "belly2", 2, False, 0, "gutted_550 (0.11 g/in)", show_enable=True),
            LayerRow(belly_frame, "belly3", 3, False, 0, "gutted_550 (0.11 g/in)", show_enable=True),
        ]

        core_frame = ttk.LabelFrame(main, text="Core Groups (listed order = cut order)", padding=8)
        core_frame.pack(fill="x", pady=4)

        ttk.Label(core_frame, text="Enabled").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Label(core_frame, text="Count").grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(core_frame, text="Material").grid(row=0, column=2, sticky="w", padx=4)
        ttk.Label(core_frame, text="Custom mass").grid(row=0, column=3, sticky="w", padx=4)

        self.core_rows = [
            LayerRow(core_frame, "core1", 1, True, 12, "gutted_550 (0.11 g/in)", show_enable=True),
            LayerRow(core_frame, "core2", 2, False, 0, "gutted_550 (0.11 g/in)", show_enable=True),
            LayerRow(core_frame, "core3", 3, False, 0, "gutted_550 (0.11 g/in)", show_enable=True),
        ]

        ttk.Label(
            core_frame,
            text="Put shot-loaded core groups earlier. Put the final continuation core group last.",
        ).grid(row=4, column=0, columnspan=4, sticky="w", padx=4, pady=(8, 0))

        self.summary_var = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.summary_var, padding=(0, 8, 0, 8), justify="left").pack(anchor="w")

        table_frame = ttk.Frame(main)
        table_frame.pack(fill="both", expand=True)

        columns = (
            "idx",
            "move",
            "node",
            "ndelta",
            "inode",
            "idelta",
            "bnode",
            "bdelta",
            "bnode_txt",
            "bdelta_txt",
        )
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        self.tree.heading("idx", text="Row #")
        self.tree.heading("move", text="Move / State")
        self.tree.heading("node", text="Normalized Node")
        self.tree.heading("ndelta", text="Normalized Delta")
        self.tree.heading("inode", text="Exact Inch Node")
        self.tree.heading("idelta", text="Exact Delta Inch")
        self.tree.heading("bnode", text="Build Node")
        self.tree.heading("bdelta", text="Build Delta")
        self.tree.heading("bnode_txt", text='Build Node (1/8")')
        self.tree.heading("bdelta_txt", text='Build Delta (1/8")')

        self.tree.column("idx", width=70, anchor="center")
        self.tree.column("move", width=420, anchor="w")
        self.tree.column("node", width=120, anchor="center")
        self.tree.column("ndelta", width=120, anchor="center")
        self.tree.column("inode", width=115, anchor="center")
        self.tree.column("idelta", width=115, anchor="center")
        self.tree.column("bnode", width=105, anchor="center")
        self.tree.column("bdelta", width=105, anchor="center")
        self.tree.column("bnode_txt", width=120, anchor="center")
        self.tree.column("bdelta_txt", width=120, anchor="center")

        yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        xscroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")

        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

    def load_beginner_preset(self) -> None:
        self.thong_length_var.set("72")
        self.family_var.set("impedance_gradient")
        self.braided_factor_var.set(f"{math.sqrt(2.0):.9f}")
        self.end_var.set("2")
        self.build_quantum_var.set(f"{BUILD_QUANTUM_IN:.3f}")
        self.build_min_delta_var.set(f"{BUILD_MIN_DELTA_IN:.3f}")
        self.drop_style_var.set("single")

        self.overlay_row.set_values(True, 12, "gutted_550 (0.11 g/in)")
        self.belly_rows[0].set_values(False, 0, "gutted_550 (0.11 g/in)")
        self.belly_rows[1].set_values(False, 0, "gutted_550 (0.11 g/in)")
        self.belly_rows[2].set_values(False, 0, "gutted_550 (0.11 g/in)")

        self.core_rows[0].set_values(True, 12, "gutted_550 (0.11 g/in)")
        self.core_rows[1].set_values(False, 0, "gutted_550 (0.11 g/in)")
        self.core_rows[2].set_values(False, 0, "gutted_550 (0.11 g/in)")

    def _read_inputs(self):
        try:
            thong_length_in = float(self.thong_length_var.get().strip())
        except Exception as exc:
            raise ValueError("Thong length must be a number.") from exc

        try:
            braided_factor = float(self.braided_factor_var.get().strip())
        except Exception as exc:
            raise ValueError("Braided factor must be a number.") from exc

        try:
            overlay_end = int(self.end_var.get().strip())
        except Exception as exc:
            raise ValueError("End value must be 2 or 4.") from exc

        try:
            build_quantum_in = float(self.build_quantum_var.get().strip())
        except Exception as exc:
            raise ValueError("Build quantum must be a number.") from exc

        try:
            build_min_delta_in = float(self.build_min_delta_var.get().strip())
        except Exception as exc:
            raise ValueError("Minimum build move must be a number.") from exc

        drop_style = self.drop_style_var.get().strip()

        overlay = self.overlay_row.to_config("overlay", braided=True, force_enabled=True)
        bellies = [
            self.belly_rows[0].to_config("belly1", braided=True),
            self.belly_rows[1].to_config("belly2", braided=True),
            self.belly_rows[2].to_config("belly3", braided=True),
        ]
        core_groups = [
            self.core_rows[0].to_config("core1", braided=False),
            self.core_rows[1].to_config("core2", braided=False),
            self.core_rows[2].to_config("core3", braided=False),
        ]

        return (
            thong_length_in,
            self.family_var.get().strip(),
            braided_factor,
            overlay_end,
            build_quantum_in,
            build_min_delta_in,
            drop_style,
            overlay,
            bellies,
            core_groups,
        )

    def calculate(self) -> None:
        try:
            (
                thong_length_in,
                family,
                braided_factor,
                overlay_end,
                build_quantum_in,
                build_min_delta_in,
                drop_style,
                overlay,
                bellies,
                core_groups,
            ) = self._read_inputs()

            if thong_length_in <= 0.0:
                raise ValueError("Thong length must be positive.")
            if braided_factor <= 1.0:
                raise ValueError("Braided factor should be > 1.0.")
            if overlay_end not in {2, 4}:
                raise ValueError("End value must be 2 or 4.")
            if family not in TAPER_OPTIONS:
                raise ValueError("Invalid taper family.")
            if build_quantum_in <= 0.0:
                raise ValueError("Build quantum must be positive.")
            if drop_style not in DROP_STYLE_OPTIONS:
                raise ValueError("Invalid drop style.")

            self.rows = build_table_rows(
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

            for item in self.tree.get_children():
                self.tree.delete(item)

            for row in self.rows:
                self.tree.insert(
                    "",
                    "end",
                    values=(
                        row["row_index"],
                        row["move"],
                        f"{row['normalized_node']:.9f}",
                        f"{row['normalized_delta']:.9f}",
                        f"{row['inch_node']:.6f}",
                        f"{row['inch_delta']:.6f}",
                        f"{row['build_node']:.6f}",
                        f"{row['build_delta']:.6f}",
                        row["build_node_text"],
                        row["build_delta_text"],
                    ),
                )

            norm_sum = sum(r["normalized_delta"] for r in self.rows)
            inch_sum = sum(r["inch_delta"] for r in self.rows)
            build_sum = sum(r["build_delta"] for r in self.rows)
            min_exact = min(r["inch_delta"] for r in self.rows)
            min_build = min(r["build_delta"] for r in self.rows)

            self.summary_var.set(
                f"{len(self.rows)} rows | "
                f"Taper: {family} | "
                f"Drop style: {drop_style} | "
                f"Thong length: {thong_length_in:.3f} in | "
                f"End at: {overlay_end}-0\n"
                f"Normalized delta sum: {norm_sum:.9f} | "
                f"Exact delta inch sum: {inch_sum:.6f} | "
                f"Build delta inch sum: {build_sum:.6f}\n"
                f"Build quantum: {build_quantum_in:.3f} in | "
                f"Requested min build move: {build_min_delta_in:.3f} in | "
                f"Smallest exact move: {min_exact:.6f} in | "
                f"Smallest build move: {min_build:.6f} in"
            )

        except Exception as exc:
            messagebox.showerror("Calculation error", str(exc))

    def export_csv(self) -> None:
        if not self.rows:
            return

        path = filedialog.asksaveasfilename(
            title="Save taper plan CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="dynamic_whip_taper_plan.csv",
        )
        if not path:
            return

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "row_index",
                "move_or_state",
                "normalized_node",
                "normalized_delta",
                "exact_inch_node",
                "exact_delta_inch",
                "build_node",
                "build_delta",
                "build_node_text",
                "build_delta_text",
                "mu_g_per_in",
            ])
            for row in self.rows:
                writer.writerow([
                    row["row_index"],
                    row["move"],
                    row["normalized_node"],
                    row["normalized_delta"],
                    row["inch_node"],
                    row["inch_delta"],
                    row["build_node"],
                    row["build_delta"],
                    row["build_node_text"],
                    row["build_delta_text"],
                    row["mu_g_per_in"],
                ])


if __name__ == "__main__":
    app = DynamicPlannerApp()
    app.mainloop()