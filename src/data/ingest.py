"""Data ingestion utilities for the NEPA dashboard."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

DATE_COLUMNS_PROJECTS = {
    "start_year": "year",
}

DATE_COLUMNS_NEPA = {
    "scoping_start": "start",
    "scoping_end": "end",
    "NOA_DEIS_date": "point",
    "DEIS_comment_open": "start",
    "DEIS_comment_close": "end",
    "NOA_FEIS_date": "point",
    "FEIS_comment_period_end": "end",
    "ROD_date": "end",
    "NOI_date": "point",
    "FEIS_comment_precision": "precision",
}

DATE_COLUMNS_DELAYS = {
    "start_date": "start",
    "end_date": "end",
}

DATE_PRECISION_SUFFIX = "_precision"
IMPUTED_SUFFIX = "_imputed"


def _resolve_precision_column(column: str, available: Iterable[str]) -> Optional[str]:
    """Return the matching precision column for a date field, if present."""

    candidates = [f"{column}{DATE_PRECISION_SUFFIX}"]
    if column.endswith("_date"):
        candidates.append(f"{column[:-5]}{DATE_PRECISION_SUFFIX}")
    if column.endswith("_period_end"):
        candidates.append(f"{column[: -len('_period_end')]}{DATE_PRECISION_SUFFIX}")
    if column.endswith("_period_start"):
        candidates.append(f"{column[: -len('_period_start')]}{DATE_PRECISION_SUFFIX}")
    if column.endswith("_start"):
        candidates.append(f"{column[:-6]}{DATE_PRECISION_SUFFIX}")
    if column.endswith("_end"):
        candidates.append(f"{column[:-4]}{DATE_PRECISION_SUFFIX}")

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in available:
            return candidate
    return None


@dataclass
class LoadedData:
    mines: pd.DataFrame
    projects: pd.DataFrame
    actions: pd.DataFrame
    delays: pd.DataFrame
    permits: pd.DataFrame
    comments: pd.DataFrame


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == "bool":
        return series
    return series.fillna(False).astype(bool)


def _get_precision(row: pd.Series, column: str) -> str:
    precision_col = f"{column}{DATE_PRECISION_SUFFIX}"
    if precision_col in row and pd.notna(row[precision_col]):
        return str(row[precision_col]).lower()
    return "day"


def _impute_date(value: pd.Timestamp | float | str | None, precision: str, *, is_start: bool) -> Tuple[pd.Timestamp | None, bool]:
    if pd.isna(value):
        return None, False
    if isinstance(value, str):
        value = value.strip()
    if value == "":
        return None, False
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None, False
    precision = precision.lower()
    imputed = False
    if precision in {"day", "exact"}:
        return parsed.normalize(), imputed
    if precision == "month":
        imputed = True
        if is_start:
            parsed = parsed.to_period("M").to_timestamp("M", "start")
        else:
            parsed = parsed.to_period("M").to_timestamp("M", "end")
        return parsed.normalize(), imputed
    if precision == "year":
        imputed = True
        year = parsed.year
        if is_start:
            parsed = pd.Timestamp(year=year, month=1, day=1)
        else:
            parsed = pd.Timestamp(year=year, month=12, day=31)
        return parsed, imputed
    return parsed.normalize(), imputed


def _apply_date_imputations(df: pd.DataFrame, columns: Iterable[str], *, start_columns: Iterable[str] = ()) -> pd.DataFrame:
    df = df.copy()
    start_columns = set(start_columns)
    precision_columns = set(df.columns)
    for column in columns:
        precision_col = _resolve_precision_column(column, precision_columns)
        is_start = column in start_columns
        imputed_flags: List[bool] = []
        values: List[pd.Timestamp | None] = []
        for _, row in df.iterrows():
            if precision_col is None:
                precision = "day"
            else:
                precision = (
                    str(row.get(precision_col, "day") or "day").strip().lower()
                )
            value, imputed = _impute_date(row.get(column), precision, is_start=is_start)
            values.append(value)
            imputed_flags.append(imputed)
        df[column] = values
        df[f"{column}{IMPUTED_SUFFIX}"] = imputed_flags
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: c.strip())


def _load_excel_tab(xl: pd.ExcelFile, name: str) -> pd.DataFrame:
    try:
        df = xl.parse(name)
    except ValueError:
        return pd.DataFrame()
    return _normalize_columns(df)


def load_workbook(path: str | Path) -> LoadedData:
    path = Path(path)
    xl = pd.ExcelFile(path)
    mines = _load_excel_tab(xl, "Mines")
    projects = _load_excel_tab(xl, "Projects")
    actions = _load_excel_tab(xl, "NEPA_Actions")
    delays = _load_excel_tab(xl, "Delays")
    permits = _load_excel_tab(xl, "Permits")
    comments = _load_excel_tab(xl, "Comments_Summary")

    # Normalize data types
    bool_cols = [
        "is_active",
        "is_supplemental",
        "litigation_flag",
        "primary",
        "on_federal_land",
    ]
    for df in (projects, actions, delays, permits):
        for col in bool_cols:
            if col in df.columns:
                df[col] = _coerce_bool(df[col])

    if "cooperating_agencies_list" in actions.columns:
        actions["cooperating_agencies_list"] = (
            actions["cooperating_agencies_list"].fillna("").astype(str)
        )

    # Apply date imputations according to precision rules
    if not actions.empty:
        # Only treat true "start" style fields as starts when applying
        # precision-based imputations. Milestone announcement dates such as
        # NOA_DEIS or NOA_FEIS are treated as points in time, so month-level
        # precision should resolve to the *end* of the month. Otherwise the
        # downstream step durations (e.g., step B: scoping_end → NOA_DEIS)
        # become negative when both fall in the same month.
        step_start_cols = [
            "scoping_start",
            "DEIS_comment_open",
            "NOI_date",
        ]
        actions = _apply_date_imputations(
            actions,
            [
                "scoping_start",
                "scoping_end",
                "NOA_DEIS_date",
                "DEIS_comment_open",
                "DEIS_comment_close",
                "NOA_FEIS_date",
                "FEIS_comment_period_end",
                "ROD_date",
                "NOI_date",
            ],
            start_columns=step_start_cols,
        )

    if not delays.empty:
        delays = _apply_date_imputations(
            delays,
            ["start_date", "end_date"],
            start_columns=["start_date"],
        )

    if not permits.empty:
        permits = _apply_date_imputations(
            permits,
            ["issue_date", "decision_date", "litigation_start", "litigation_end"],
            start_columns=["issue_date", "litigation_start"],
        )

    return LoadedData(
        mines=mines,
        projects=projects,
        actions=actions,
        delays=delays,
        permits=permits,
        comments=comments,
    )


def to_json_records(df: pd.DataFrame) -> str:
    """Serialize a DataFrame to JSON records for Dash stores."""

    if df.empty:
        return "[]"
    return df.to_json(orient="records", date_format="iso")
