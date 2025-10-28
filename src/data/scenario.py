"""Scenario engine for NEPA timelines."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from src.utils.config import NepaConfig


@dataclass
class ScenarioParameters:
    scenario: str = "baseline"
    eis_cap_days: int | None = None
    ea_cap_days: int | None = None
    litigation_scale: float | None = None
    rework_scale: float | None = None
    policy_change_scale: float | None = None
    reuse_scale: float | None = None
    compress_steps: Iterable[str] | None = None


def _with_overrides(config: NepaConfig, params: ScenarioParameters) -> NepaConfig:
    overrides = {}
    for field_name in (
        "eis_cap_days",
        "ea_cap_days",
        "litigation_scale",
        "rework_scale",
        "policy_change_scale",
        "reuse_scale",
        "compressible_steps",
    ):
        value = getattr(params, field_name if field_name != "compressible_steps" else "compress_steps")
        if value is not None:
            overrides[field_name] = value
    return replace(config, **overrides)


def apply_scenario(
    action_steps: pd.DataFrame,
    action_summary: pd.DataFrame,
    delays: pd.DataFrame,
    params: ScenarioParameters,
    config: NepaConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return scenario-adjusted copies of the step and summary tables."""

    scenario_config = _with_overrides(config, params)
    steps = action_steps.copy()
    summary = action_summary.copy()
    if steps.empty or summary.empty:
        return steps, summary

    steps["scenario_days"] = steps["days"].astype(float)
    steps["scenario_delay_days_total"] = steps["delay_days_total"].astype(float)
    summary["scenario_total_days"] = summary["total_days"].astype(float)
    summary["scenario_delay_days_total"] = summary["delay_days_total"].astype(float)
    summary["scenario_meets_cap"] = summary["meets_cap"]
    summary["scenario_over_cap_days"] = summary["over_cap_days"]

    scenario_name = params.scenario.lower()

    if scenario_name == "trump2020":
        steps, summary = _apply_trump2020(steps, summary, scenario_config)
    elif scenario_name == "speed":
        steps, summary = _apply_speed_act(steps, summary, delays, scenario_config)

    return steps, summary


def _apply_trump2020(
    steps: pd.DataFrame,
    summary: pd.DataFrame,
    config: NepaConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    compress_steps = list(config.compressible_steps)
    for nepa_id, summary_row in summary.iterrows():
        if summary_row.get("action_level") != "EIS":
            continue
        noi = summary_row.get("noi_date") or summary_row.get("window_start")
        rod = summary_row.get("rod_date") or summary_row.get("window_end")
        if pd.isna(noi) or pd.isna(rod):
            continue
        cap = config.eis_cap_days
        if pd.isna(cap):
            continue
        cap_window_days = (rod - noi).days + 1
        if cap_window_days <= cap:
            continue
        reduction_needed = cap_window_days - cap
        step_mask = (steps["nepa_id"] == summary_row["nepa_id"]) & (steps["step"].isin(compress_steps))
        compressible_days = steps.loc[step_mask, "scenario_days"].sum()
        if compressible_days <= 0:
            continue
        for idx in steps.loc[step_mask].index:
            portion = steps.at[idx, "scenario_days"] / compressible_days
            reduction = reduction_needed * portion
            new_days = max(0.0, steps.at[idx, "scenario_days"] - reduction)
            steps.at[idx, "scenario_days"] = new_days
            if compressible_days > 0:
                delay_share = steps.at[idx, "scenario_delay_days_total"] / compressible_days
                steps.at[idx, "scenario_delay_days_total"] = max(
                    0.0,
                    steps.at[idx, "scenario_delay_days_total"] - reduction * delay_share,
                )
        summary_idx = summary_row.name
        new_total = steps.loc[steps["nepa_id"] == summary_row["nepa_id"], "scenario_days"].sum()
        summary.at[summary_idx, "scenario_total_days"] = new_total
        summary.at[summary_idx, "scenario_delay_days_total"] = steps.loc[
            steps["nepa_id"] == summary_row["nepa_id"], "scenario_delay_days_total"
        ].sum()
        summary.at[summary_idx, "scenario_meets_cap"] = True
        summary.at[summary_idx, "scenario_over_cap_days"] = max(0.0, new_total - cap)
    return steps, summary


def _apply_speed_act(
    steps: pd.DataFrame,
    summary: pd.DataFrame,
    delays: pd.DataFrame,
    config: NepaConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if delays.empty:
        return steps, summary

    scale_map = {
        "Modeling_Rework": config.rework_scale,
        "Policy_Change": config.policy_change_scale,
        "Litigation": config.litigation_scale,
    }
    reuse_steps = list(config.compressible_steps)

    delay_grouped = delays.groupby("nepa_id")
    for summary_idx, summary_row in summary.iterrows():
        nepa_id = summary_row["nepa_id"]
        if nepa_id not in delay_grouped.groups:
            continue
        noi = summary_row.get("noi_date") or summary_row.get("window_start")
        window_start = summary_row.get("window_start")
        window_end = summary_row.get("window_end")
        action_steps = steps[steps["nepa_id"] == nepa_id]
        step_lookup = {row.step: row for row in action_steps.itertuples()}
        reductions: Dict[str, float] = {idx: 0.0 for idx in action_steps.index}
        delay_subset = delay_grouped.get_group(nepa_id)
        for _, delay in delay_subset.iterrows():
            category = delay.get("category")
            scale = scale_map.get(category)
            if scale is None:
                continue
            if category in {"Modeling_Rework", "Policy_Change"}:
                start = delay.get("start_date")
                if noi is not None and start is not None and start < noi:
                    continue
            delay_start = delay.get("start_date")
            delay_end = delay.get("end_date")
            if pd.isna(delay_start) or pd.isna(delay_end):
                continue
            for idx, step_row in action_steps.iterrows():
                step_start = step_row.get("start_date") or window_start
                step_end = step_row.get("end_date") or window_end
                if step_start is None or step_end is None:
                    continue
                overlap_start = max(step_start, delay_start)
                overlap_end = min(step_end, delay_end)
                if overlap_start > overlap_end:
                    continue
                days = (overlap_end - overlap_start).days + 1
                reduction = days * (1 - scale)
                reductions[idx] += reduction
        for idx, reduction in reductions.items():
            if reduction <= 0:
                continue
            current = steps.at[idx, "scenario_days"]
            steps.at[idx, "scenario_days"] = max(0.0, current - reduction)
            delay_current = steps.at[idx, "scenario_delay_days_total"]
            steps.at[idx, "scenario_delay_days_total"] = max(0.0, delay_current - reduction)
        for idx, row in action_steps.iterrows():
            if row["step"] in reuse_steps:
                steps.at[idx, "scenario_days"] = max(0.0, steps.at[idx, "scenario_days"] * config.reuse_scale)
        summary.at[summary_idx, "scenario_total_days"] = steps.loc[
            steps["nepa_id"] == nepa_id, "scenario_days"
        ].sum()
        summary.at[summary_idx, "scenario_delay_days_total"] = steps.loc[
            steps["nepa_id"] == nepa_id, "scenario_delay_days_total"
        ].sum()
        level = summary_row.get("action_level")
        cap = config.cap_lookup.get(level)
        if cap is not None:
            total = summary.at[summary_idx, "scenario_total_days"]
            summary.at[summary_idx, "scenario_meets_cap"] = total <= cap
            summary.at[summary_idx, "scenario_over_cap_days"] = max(0.0, total - cap)
    return steps, summary
