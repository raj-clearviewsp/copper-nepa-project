"""Business rule transformations for NEPA analytics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.ingest import LoadedData
from src.utils.config import NepaConfig

STEP_LABELS = {
    "EIS": ["A", "B", "C", "D", "E"],
    "EA": ["A", "B", "C", "D", "E"],
}


@dataclass
class ActionStep:
    nepa_id: str
    project_id: str
    action_level: str
    step: str
    start_date: Optional[pd.Timestamp]
    end_date: Optional[pd.Timestamp]
    days: Optional[int]
    days_in_window: Optional[int]
    delay_days_total: float
    delay_days_by_category: Dict[str, float]
    is_flagged: bool


@dataclass
class ActionSummary:
    nepa_id: str
    project_id: str
    action_level: str
    mine_id: Optional[str]
    total_days: Optional[int]
    window_days: Optional[int]
    delay_days_total: float
    meets_cap: Optional[bool]
    cap_days: Optional[int]
    over_cap_days: Optional[int]
    cohort_year: Optional[int]
    cohort_decade: Optional[str]
    had_litigation: bool
    is_complete: bool
    flag_reason: str
    window_start: Optional[pd.Timestamp]
    window_end: Optional[pd.Timestamp]
    noi_date: Optional[pd.Timestamp]
    rod_date: Optional[pd.Timestamp]
    ea_start: Optional[pd.Timestamp]
    fonzi_date: Optional[pd.Timestamp]
    state: Optional[str]
    lead_agency_id: Optional[str]
    mine_type: Optional[str]
    tailings_type: Optional[str]
    on_federal_land: Optional[bool]
    cooperating_agencies_count: Optional[float]
    cooperating_agencies_list: Optional[str]
    litigation_flag: Optional[bool]
    status: Optional[str]


def _daterange(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    if start is None or end is None:
        return []
    if start > end:
        return []
    return list(pd.date_range(start, end, freq="D"))


def _compute_step_days(start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> Optional[int]:
    if start is None or end is None or pd.isna(start) or pd.isna(end):
        return None
    if end < start:
        return None
    return int((end - start).days) + 1


def _safe_max(values: Iterable[Optional[pd.Timestamp]]) -> Optional[pd.Timestamp]:
    valid = [v for v in values if v is not None and not pd.isna(v)]
    if not valid:
        return None
    return max(valid)


def _safe_min(values: Iterable[Optional[pd.Timestamp]]) -> Optional[pd.Timestamp]:
    valid = [v for v in values if v is not None and not pd.isna(v)]
    if not valid:
        return None
    return min(valid)


def _assign_cohort_year(value: Optional[pd.Timestamp]) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    return int(value.year)


def _assign_cohort_decade(year: Optional[int]) -> Optional[str]:
    if year is None:
        return None
    decade_start = year - (year % 10)
    return f"{decade_start}s"


def _build_steps_for_action(row: pd.Series) -> Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]]:
    level = str(row.get("action_level", "EIS") or "EIS").upper()
    steps: Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]] = {}

    scoping_start = row.get("scoping_start")
    scoping_end = row.get("scoping_end") or scoping_start
    noa_deis = row.get("NOA_DEIS_date")
    deis_open = row.get("DEIS_comment_open") or noa_deis
    deis_close = row.get("DEIS_comment_close") or deis_open
    noa_feis = row.get("NOA_FEIS_date")
    feis_comment_end = row.get("FEIS_comment_period_end") or noa_feis
    rod = row.get("ROD_date")

    noi = row.get("NOI_date")
    fonzi = row.get("FONSI_date") if "FONSI_date" in row else None

    if level == "EA":
        # Map analogous EA steps
        steps["A"] = (scoping_start, scoping_end)
        steps["B"] = (scoping_end, row.get("DEIS_comment_open") or row.get("Draft_EA_date") or scoping_end)
        steps["C"] = (
            row.get("DEIS_comment_open") or row.get("Draft_EA_date"),
            row.get("DEIS_comment_close") or row.get("Draft_EA_end_date") or row.get("DEIS_comment_open"),
        )
        steps["D"] = (
            steps["C"][1],
            row.get("Final_EA_date") or steps["C"][1],
        )
        steps["E"] = (steps["D"][1], row.get("FONSI_date") or row.get("ROD_date"))
    else:
        steps["A"] = (scoping_start, scoping_end)
        steps["B"] = (scoping_end, noa_deis)
        steps["C"] = (deis_open, deis_close)
        steps["D"] = (deis_close, noa_feis)
        start_e = feis_comment_end
        if start_e is None:
            start_e = noa_feis
        steps["E"] = (start_e, rod)

    return steps


def _validate_monotonicity(steps: Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]]) -> Tuple[bool, str]:
    ordered_steps = [steps[key] for key in ["A", "B", "C", "D", "E"] if key in steps]
    prev_end: Optional[pd.Timestamp] = None
    for idx, (start, end) in enumerate(ordered_steps):
        if start is None or end is None:
            continue
        if end < start:
            return False, f"Negative duration in step {chr(ord('A') + idx)}"
        if prev_end and start and start < prev_end:
            return False, "Non-monotonic step boundaries"
        if end:
            prev_end = end
    return True, ""


def _compute_delay_overlaps(
    step_range: Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]],
    delays: pd.DataFrame,
    config: NepaConfig,
) -> Tuple[float, Dict[str, float]]:
    start, end = step_range
    if start is None or end is None or start > end or delays.empty:
        return 0.0, {}

    step_days = _daterange(start, end)
    if not step_days:
        return 0.0, {}

    day_records: Dict[pd.Timestamp, List[str]] = {day: [] for day in step_days}

    for _, delay in delays.iterrows():
        delay_start = delay.get("start_date")
        delay_end = delay.get("end_date")
        if delay_start is None or delay_end is None or delay_start > delay_end:
            continue
        overlap_start = max(delay_start, start)
        overlap_end = min(delay_end, end)
        if overlap_start > overlap_end:
            continue
        category = str(delay.get("category") or "Unspecified")
        subcategory = str(delay.get("subcategory") or "Unspecified")
        key = f"{category} › {subcategory}"
        for day in pd.date_range(overlap_start, overlap_end, freq="D"):
            if day in day_records:
                if config.overlap_allocation == "primary" and not delay.get("primary", False):
                    continue
                day_records[day].append(key)

    total_unique_days = 0
    category_totals: Dict[str, float] = {}
    for day, categories in day_records.items():
        if not categories:
            continue
        total_unique_days += 1
        if config.overlap_allocation == "equal" or not categories:
            share = 1.0 / len(categories) if categories else 0
            for cat in categories:
                category_totals[cat] = category_totals.get(cat, 0.0) + share
        elif config.overlap_allocation == "primary":
            for cat in categories:
                category_totals[cat] = category_totals.get(cat, 0.0) + 1.0
        else:
            # proportional by non-overlapped share: treat each instance as full day for now
            share = 1.0 / len(categories) if categories else 0
            for cat in categories:
                category_totals[cat] = category_totals.get(cat, 0.0) + share

    return float(total_unique_days), category_totals


def _determine_litigation(
    action_row: pd.Series,
    delays: pd.DataFrame,
    permits: pd.DataFrame,
    window: Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]],
) -> bool:
    if "litigation_flag" in action_row and bool(action_row.get("litigation_flag")):
        return True
    if not delays.empty:
        if any(delay.get("category") == "Litigation" for _, delay in delays.iterrows()):
            return True
    if permits.empty:
        return False
    window_start, window_end = window
    for _, permit in permits.iterrows():
        if permit.get("litigation_flag"):
            start = permit.get("litigation_start") or permit.get("issue_date")
            end = permit.get("litigation_end") or permit.get("decision_date") or permit.get("issue_date")
            if window_start and window_end and start and end:
                if start <= window_end and end >= window_start:
                    return True
            else:
                return True
    return False


def derive_tables(data: LoadedData, config: NepaConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create tidy action-level and step-level tables along with QA flags."""

    actions = data.actions.copy()
    projects = data.projects.copy()
    mines = data.mines.copy()
    delays = data.delays.copy()
    permits = data.permits.copy()

    if actions.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    actions["action_level"] = actions["action_level"].str.upper().fillna("EIS")

    if not projects.empty:
        actions = actions.merge(projects, on="project_id", how="left", suffixes=("", "_project"))
    if not mines.empty:
        actions = actions.merge(mines, on="mine_id", how="left", suffixes=("", "_mine"))

    delays_by_nepa: Dict[str, pd.DataFrame] = {}
    if not delays.empty:
        delays_by_nepa = {key: group.copy() for key, group in delays.groupby("nepa_id")}

    permits_by_action: Dict[str, pd.DataFrame] = {}
    if not permits.empty:
        if "nepa_id" in permits.columns:
            permits_by_action = {key: group.copy() for key, group in permits.groupby("nepa_id")}
        elif "project_id" in permits.columns:
            # fall back to project join
            for key, group in permits.groupby("project_id"):
                permits_by_action[key] = group.copy()

    step_records: List[Dict[str, object]] = []
    summary_records: List[Dict[str, object]] = []
    qa_records: List[Dict[str, object]] = []

    for _, row in actions.iterrows():
        nepa_id = row.get("nepa_id")
        project_id = row.get("project_id")
        level = row.get("action_level", "EIS")
        steps = _build_steps_for_action(row)
        is_valid, reason = _validate_monotonicity(steps)
        action_delays = delays_by_nepa.get(nepa_id, pd.DataFrame())
        if action_delays.empty and project_id in permits_by_action and "nepa_id" not in permits.columns:
            action_permits = permits_by_action.get(project_id, pd.DataFrame())
        else:
            action_permits = permits_by_action.get(nepa_id, pd.DataFrame())

        step_objects: List[ActionStep] = []
        total_days = 0
        window_start = steps.get("A", (None, None))[0]
        window_end = steps.get("E", (None, None))[1]
        flag_reason = reason
        for step_label, (start, end) in steps.items():
            days = _compute_step_days(start, end)
            if days is None:
                flag_reason = flag_reason or "Missing milestone"
            else:
                total_days += days
            delay_total, delay_breakdown = _compute_delay_overlaps((start, end), action_delays, config)
            step_objects.append(
                ActionStep(
                    nepa_id=nepa_id,
                    project_id=project_id,
                    action_level=level,
                    step=step_label,
                    start_date=start,
                    end_date=end,
                    days=days,
                    days_in_window=days,
                    delay_days_total=delay_total,
                    delay_days_by_category=delay_breakdown,
                    is_flagged=not is_valid or days is None,
                )
            )

        window_days = _compute_step_days(window_start, window_end)
        cap_days = config.cap_lookup.get(level, None)
        meets_cap = None
        over_cap_days = None
        noi = row.get("NOI_date")
        if level == "EIS":
            noi_start = noi or window_start
            if noi_start is not None and window_end is not None:
                cap_window_days = _compute_step_days(noi_start, window_end)
                if cap_window_days is not None and cap_days is not None:
                    meets_cap = cap_window_days <= cap_days
                    over_cap_days = max(0, cap_window_days - cap_days)
        else:
            ea_start = row.get("scoping_start")
            fonzi = row.get("FONSI_date") or window_end
            if ea_start is not None and fonzi is not None and cap_days is not None:
                cap_window_days = _compute_step_days(ea_start, fonzi)
                if cap_window_days is not None:
                    meets_cap = cap_window_days <= cap_days
                    over_cap_days = max(0, cap_window_days - cap_days)

        had_litigation = _determine_litigation(row, action_delays, action_permits, (window_start, window_end))
        cohort_year = _assign_cohort_year(window_start)
        cohort_decade = _assign_cohort_decade(cohort_year)
        complete_statuses = {"complete", "completed", "closed", "rod", "withdrawn", "decision issued"}
        is_complete = str(row.get("status") or "").strip().lower() in complete_statuses
        if not config.include_active_actions and not is_complete:
            flag_reason = flag_reason or "Active action excluded"

        summary_records.append(
            ActionSummary(
                nepa_id=nepa_id,
                project_id=project_id,
                action_level=level,
                mine_id=row.get("mine_id"),
                total_days=total_days if total_days > 0 else None,
                window_days=window_days,
                delay_days_total=float(sum(step.delay_days_total for step in step_objects)),
                meets_cap=meets_cap,
                cap_days=cap_days,
                over_cap_days=over_cap_days,
                cohort_year=cohort_year,
                cohort_decade=cohort_decade,
                had_litigation=had_litigation,
                is_complete=is_complete,
                flag_reason=flag_reason,
                window_start=window_start,
                window_end=window_end,
                noi_date=noi,
                rod_date=window_end,
                ea_start=row.get("scoping_start"),
                fonzi_date=row.get("FONSI_date") if "FONSI_date" in row else None,
                state=row.get("state"),
                lead_agency_id=row.get("lead_agency_id"),
                mine_type=row.get("mine_type"),
                tailings_type=row.get("tailings_type"),
                on_federal_land=row.get("on_federal_land"),
                cooperating_agencies_count=row.get("cooperating_agencies_count"),
                cooperating_agencies_list=row.get("cooperating_agencies_list"),
                litigation_flag=row.get("litigation_flag"),
                status=row.get("status"),
            ).__dict__
        )

        for step in step_objects:
            record = {
                "nepa_id": step.nepa_id,
                "project_id": step.project_id,
                "action_level": step.action_level,
                "step": step.step,
                "start_date": step.start_date,
                "end_date": step.end_date,
                "days": step.days,
                "days_in_window": step.days_in_window,
                "delay_days_total": step.delay_days_total,
                "delay_days_by_category": json.dumps(step.delay_days_by_category),
                "is_flagged": step.is_flagged,
            }
            step_records.append(record)

        if flag_reason:
            qa_records.append(
                {
                    "nepa_id": nepa_id,
                    "flag_reason": flag_reason,
                }
            )

    action_steps_df = pd.DataFrame(step_records)
    action_summary_df = pd.DataFrame(summary_records)
    qa_df = pd.DataFrame(qa_records)

    return action_steps_df, action_summary_df, qa_df
