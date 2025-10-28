"""Dash callbacks powering the NEPA dashboard."""

from __future__ import annotations

import json
from dataclasses import asdict
from io import StringIO
from typing import Dict, List, Tuple

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from src.data.scenario import ScenarioParameters, apply_scenario
from src.utils.config import NepaConfig


def _df_from_store(data) -> pd.DataFrame:
    """Normalize dcc.Store payloads into DataFrames.

    Stores backed by :func:`to_json_records` now provide native Python lists of
    dictionaries, but older sessions or downstream callbacks may still hand us
    a JSON string. Handle both to make troubleshooting data issues easier.
    """

    if data is None or data == "" or data == []:
        return pd.DataFrame()
    if isinstance(data, str):
        return pd.read_json(StringIO(data), orient="records")
    return pd.DataFrame(data)


def _format_days(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{int(round(value)):,}"


def register_callbacks(app: dash.Dash, config: NepaConfig) -> None:
    @app.callback(
        Output("store-filtered", "data"),
        Input("store-action-summary", "data"),
        Input("store-action-steps", "data"),
        Input("store-delays", "data"),
        Input("store-qa", "data"),
        Input("filter-action-level", "value"),
        Input("filter-state", "value"),
        Input("filter-lead-agency", "value"),
        Input("filter-cohort-year", "value"),
        Input("filter-cohort-decade", "value"),
        Input("filter-mine-type", "value"),
        Input("filter-tailings-type", "value"),
        Input("filter-litigation", "value"),
        Input("filter-coop-count", "value"),
        Input("filter-flags", "value"),
    )
    def update_filtered_store(
        summary_json,
        steps_json,
        delays_json,
        qa_json,
        action_levels,
        states,
        agencies,
        cohort_years,
        cohort_decades,
        mine_types,
        tailings_types,
        litigation_choice,
        coop_range,
        flags,
    ):
        summary_df = _df_from_store(summary_json)
        steps_df = _df_from_store(steps_json)
        delays_df = _df_from_store(delays_json)
        qa_df = _df_from_store(qa_json)
        if summary_df.empty:
            return json.dumps({})

        mask = pd.Series(True, index=summary_df.index)

        if action_levels:
            mask &= summary_df["action_level"].isin(action_levels)
        if states:
            mask &= summary_df["state"].isin(states)
        if agencies:
            mask &= summary_df["lead_agency_id"].isin(agencies)
        if cohort_years:
            mask &= summary_df["cohort_year"].isin(cohort_years)
        if cohort_decades:
            mask &= summary_df["cohort_decade"].isin(cohort_decades)
        if mine_types:
            mask &= summary_df["mine_type"].isin(mine_types)
        if tailings_types:
            mask &= summary_df["tailings_type"].isin(tailings_types)
        if litigation_choice == "yes":
            mask &= summary_df["had_litigation"]
        elif litigation_choice == "no":
            mask &= ~summary_df["had_litigation"]
        if coop_range and len(coop_range) == 2:
            lower, upper = coop_range
            counts = summary_df["cooperating_agencies_count"].fillna(0)
            mask &= (counts >= lower) & (counts <= upper)
        if "include_flagged" not in (flags or []):
            mask &= summary_df["flag_reason"].isna() | (summary_df["flag_reason"] == "")
        if "include_active" not in (flags or []):
            mask &= summary_df["is_complete"]

        filtered_summary = summary_df[mask].copy()
        filtered_steps = steps_df[steps_df["nepa_id"].isin(filtered_summary["nepa_id"])]
        filtered_delays = delays_df[delays_df["nepa_id"].isin(filtered_summary["nepa_id"])]
        filtered_qa = qa_df[qa_df["nepa_id"].isin(filtered_summary["nepa_id"])]

        payload = {
            "summary": json.loads(filtered_summary.to_json(orient="records", date_format="iso")),
            "steps": json.loads(filtered_steps.to_json(orient="records", date_format="iso")),
            "delays": json.loads(filtered_delays.to_json(orient="records", date_format="iso")),
            "qa": json.loads(filtered_qa.to_json(orient="records", date_format="iso")),
        }
        return json.dumps(payload)

    @app.callback(
        Output("store-scenario", "data"),
        Input("store-filtered", "data"),
        Input("scenario-selector", "value"),
        Input("input-eis-cap", "value"),
        Input("input-ea-cap", "value"),
        Input("input-litigation-scale", "value"),
        Input("input-rework-scale", "value"),
        Input("input-policy-scale", "value"),
        Input("input-reuse-scale", "value"),
    )
    def update_scenario_store(
        filtered_json,
        scenario_value,
        eis_cap,
        ea_cap,
        litigation_scale,
        rework_scale,
        policy_scale,
        reuse_scale,
    ):
        if not filtered_json:
            return json.dumps({})
        payload = json.loads(filtered_json)
        steps_df = pd.DataFrame(payload.get("steps", []))
        summary_df = pd.DataFrame(payload.get("summary", []))
        delays_df = pd.DataFrame(payload.get("delays", []))
        if summary_df.empty:
            return json.dumps({})
        params = ScenarioParameters(
            scenario=scenario_value,
            eis_cap_days=eis_cap,
            ea_cap_days=ea_cap,
            litigation_scale=litigation_scale,
            rework_scale=rework_scale,
            policy_change_scale=policy_scale,
            reuse_scale=reuse_scale,
        )
        scenario_steps, scenario_summary = apply_scenario(steps_df, summary_df, delays_df, params, config)
        scenario_payload = {
            "steps": json.loads(scenario_steps.to_json(orient="records", date_format="iso")),
            "summary": json.loads(scenario_summary.to_json(orient="records", date_format="iso")),
            "params": asdict(params),
        }
        return json.dumps(scenario_payload)

    @app.callback(
        Output("kpi-total-days", "children"),
        Output("kpi-delay-days", "children"),
        Output("kpi-litigation", "children"),
        Output("kpi-cap", "children"),
        Output("kpi-total-days-context", "children"),
        Output("kpi-delay-days-context", "children"),
        Output("kpi-litigation-context", "children"),
        Output("kpi-cap-context", "children"),
        Output("timeline-story", "children"),
        Output("fig-waterfall", "figure"),
        Output("waterfall-annotation", "children"),
        Output("fig-variance", "figure"),
        Output("variance-annotation", "children"),
        Output("table-step-summary", "data"),
        Output("fig-delay-pareto", "figure"),
        Output("pareto-annotation", "children"),
        Output("table-affected-actions", "data"),
        Output("fig-agency-benchmark", "figure"),
        Output("fig-cooperation", "figure"),
        Output("fig-litigation", "figure"),
        Output("litigation-annotation", "children"),
        Output("fig-copper", "figure"),
        Output("copper-summary", "children"),
        Output("fig-distribution", "figure"),
        Output("table-top-movers", "data"),
        Output("scenario-assumptions", "children"),
        Output("table-qa", "data"),
        Input("store-filtered", "data"),
        Input("store-scenario", "data"),
        Input("store-projects", "data"),
        Input("store-mines", "data"),
        Input("store-permits", "data"),
        Input("fig-delay-pareto", "clickData"),
    )
    def update_visuals(
        filtered_json,
        scenario_json,
        projects_json,
        mines_json,
        permits_json,
        pareto_click,
    ):
        empty_fig = go.Figure()
        if not filtered_json:
            return (
                "—",
                "—",
                "—",
                "—",
                "N=0",
                "",
                "",
                "",
                html.Div("No completed actions in view.", className="insight-body"),
                empty_fig,
                html.Div("No data available for the selected filters.", className="insight-body"),
                empty_fig,
                "",
                [],
                empty_fig,
                "",
                [],
                empty_fig,
                empty_fig,
                empty_fig,
                html.Div("No permit view for current filters.", className="insight-body"),
                empty_fig,
                "",
                empty_fig,
                [],
                "",
                [],
            )
        payload = json.loads(filtered_json)
        summary_df = pd.DataFrame(payload.get("summary", []))
        steps_df = pd.DataFrame(payload.get("steps", []))
        delays_df = pd.DataFrame(payload.get("delays", []))
        qa_df = pd.DataFrame(payload.get("qa", []))

        scenario_payload = json.loads(scenario_json) if scenario_json else {}
        scenario_steps = pd.DataFrame(scenario_payload.get("steps", []))
        scenario_summary = pd.DataFrame(scenario_payload.get("summary", []))
        params = scenario_payload.get("params", {"scenario": "baseline"})

        projects_df = _df_from_store(projects_json)
        mines_df = _df_from_store(mines_json)
        permits_df = _df_from_store(permits_json)
        if not permits_df.empty:
            if "nepa_id" in permits_df.columns:
                permits_df = permits_df[permits_df["nepa_id"].isin(summary_df["nepa_id"])]
            elif "project_id" in permits_df.columns:
                permits_df = permits_df[permits_df["project_id"].isin(summary_df["project_id"])]

        timeline_story = html.Div("No completed actions in view.", className="insight-body")
        waterfall_note = html.Div(
            "No step data available for the current slice.", className="insight-body"
        )
        variance_note = html.Div(
            "Insufficient data to assess variability.", className="insight-body"
        )
        pareto_note = html.Div(
            "No recorded delays in the NEPA window for these actions.", className="insight-body"
        )
        litigation_annotation = html.Div(
            "No permit view for current filters.", className="insight-body"
        )
        pareto_fig = empty_fig
        affected_data: List[Dict[str, object]] = []
        top_step = None
        top_step_days = None
        delay_focus_step = None
        delay_focus_days = 0

        n_actions = len(summary_df)
        median_total = summary_df["total_days"].median() if not summary_df.empty else np.nan
        median_delay = summary_df["delay_days_total"].median() if not summary_df.empty else np.nan
        litigation_pct = (
            (summary_df["had_litigation"].mean() * 100) if not summary_df.empty else np.nan
        )
        cap_series = summary_df["meets_cap"].dropna()
        cap_pct = (cap_series.mean() * 100) if not cap_series.empty else np.nan

        kpi_total = _format_days(median_total)
        kpi_delay = _format_days(median_delay)
        kpi_lit = f"{litigation_pct:.1f}%" if not np.isnan(litigation_pct) else "—"
        kpi_cap = f"{cap_pct:.1f}%" if not np.isnan(cap_pct) else "—"
        kpi_total_ctx = f"N={n_actions} actions"
        if n_actions < 10:
            kpi_total_ctx += " • Warning: N<10"
        kpi_delay_ctx = "Median in-window delay days"
        kpi_lit_ctx = "Share of actions with litigation"
        kpi_cap_ctx = "Share meeting cap (EIS 2y / EA 1y)"

        # Waterfall computation
        if steps_df.empty:
            waterfall_fig = empty_fig
            step_summary = []
            variance_fig = empty_fig
        else:
            step_stats = (
                steps_df.groupby("step")
                .agg(
                    median_days=("days", "median"),
                    p10=("days", lambda x: np.percentile(x.dropna(), 10) if len(x.dropna()) else np.nan),
                    p90=("days", lambda x: np.percentile(x.dropna(), 90) if len(x.dropna()) else np.nan),
                    delay_median=("delay_days_total", "median"),
                    N=("days", lambda x: x.notna().sum()),
                )
                .reindex(["A", "B", "C", "D", "E"])
                .dropna(how="all")
            )
            waterfall_fig = go.Figure()
            waterfall_fig.add_bar(
                x=step_stats.index,
                y=step_stats["median_days"],
                name="Median days",
                marker_color="#8C6C4F",
            )
            waterfall_fig.add_bar(
                x=step_stats.index,
                y=step_stats["delay_median"],
                name="Median delay days",
                marker_pattern_shape="/",
                marker_color="#C49A6C",
            )
            waterfall_fig.update_layout(
                barmode="stack",
                template="plotly_white",
                legend_orientation="h",
                legend_yanchor="bottom",
                legend_y=1.02,
                legend_x=0,
                margin=dict(t=30),
                yaxis_title="Days",
            )
            total_step_median = step_stats["median_days"].fillna(0).sum()
            delay_median_total = step_stats["delay_median"].fillna(0).sum()
            if not step_stats["median_days"].dropna().empty:
                top_step = step_stats["median_days"].astype(float).idxmax()
                top_step_days = step_stats.loc[top_step, "median_days"]
            if not step_stats["delay_median"].dropna().empty:
                delay_focus_step = step_stats["delay_median"].astype(float).idxmax()
                delay_focus_days = step_stats.loc[delay_focus_step, "delay_median"]
            bullets = []
            if top_step is not None and pd.notna(top_step_days):
                bullets.append(
                    html.Li(
                        f"Step {top_step} is longest at {int(round(top_step_days)):,} median days.",
                    )
                )
            if (
                delay_focus_step is not None
                and pd.notna(delay_focus_days)
                and delay_focus_days > 0
            ):
                bullets.append(
                    html.Li(
                        f"Delays peak in Step {delay_focus_step}, adding {int(round(delay_focus_days)):,} days to the median action.",
                    )
                )
            bullets.append(html.Li(f"Sample size: N={n_actions} actions."))
            waterfall_note = html.Div(
                [
                    html.P(
                        [
                            html.Span(
                                f"{int(round(total_step_median)):,} days",
                                className="insight-highlight",
                            ),
                            " median span across steps A–E.",
                            html.Br(),
                            html.Span(
                                f"{int(round(delay_median_total)):,} days",
                                className="insight-highlight",
                            ),
                            " attributable to in-window delays.",
                        ],
                        className="insight-lead",
                    ),
                    html.Ul(bullets, className="insight-list"),
                ],
                className="insight-body",
            )

            variance_df = steps_df.dropna(subset=["days"]).copy()
            variance_df["step"] = variance_df["step"].astype(str)
            variance_fig = px.violin(
                variance_df,
                x="step",
                y="days",
                box=True,
                points="outliers",
                color="step",
                color_discrete_sequence=["#4C6B75", "#5D7F86", "#6F9397", "#81A7A8", "#93BBB9"],
            )
            variance_fig.update_layout(showlegend=False, template="plotly_white", yaxis_title="Days")
            summary_stats = []
            variance_annotations = []
            for step, group in variance_df.groupby("step"):
                stats = {
                    "step": step,
                    "median_days": round(group["days"].median(), 1),
                    "p10": round(np.percentile(group["days"], 10), 1),
                    "p90": round(np.percentile(group["days"], 90), 1),
                    "N": int(group["days"].count()),
                }
                summary_stats.append(stats)
                cv = group["days"].std(ddof=0) / group["days"].mean() if group["days"].mean() else np.nan
                variance_annotations.append((step, cv))
            valid = [item for item in variance_annotations if not np.isnan(item[1])]
            if valid:
                highest = max(valid, key=lambda x: x[1])
                lowest = min(valid, key=lambda x: x[1])
                variance_note = html.Div(
                    [
                        html.P(
                            "Predictability differs sharply by milestone stage.",
                            className="insight-lead",
                        ),
                        html.Ul(
                            [
                                html.Li(
                                    f"Step {highest[0]} shows the widest variability (CV {highest[1]:.2f})."
                                ),
                                html.Li(
                                    f"Step {lowest[0]} is most predictable (CV {lowest[1]:.2f})."
                                ),
                                html.Li(f"Coverage: N={n_actions} actions."),
                            ],
                            className="insight-list",
                        ),
                    ],
                    className="insight-body",
                )
            else:
                variance_note = html.Div(
                    "Not enough variability data for this slice.", className="insight-body"
                )
            step_summary = summary_stats
        # Delay Pareto
        if steps_df.empty:
            pareto_fig = empty_fig
            affected_data = []
        else:
            delay_records: List[Tuple[str, float]] = []
            for _, row in steps_df.iterrows():
                if pd.isna(row.get("delay_days_by_category")):
                    continue
                categories = json.loads(row["delay_days_by_category"])
                for cat, value in categories.items():
                    delay_records.append((cat, float(value)))
            delay_df = (
                pd.DataFrame(delay_records, columns=["category", "days"])
                if delay_records
                else pd.DataFrame(columns=["category", "days"])
            )
            if delay_df.empty:
                pareto_fig = empty_fig
                affected_data = []
            else:
                pareto_df = delay_df.groupby("category", as_index=False)["days"].sum().sort_values("days", ascending=False)
                pareto_df["cum_share"] = pareto_df["days"].cumsum() / pareto_df["days"].sum()
                pareto_fig = go.Figure()
                pareto_fig.add_bar(x=pareto_df["category"], y=pareto_df["days"], name="Delay days", marker_color="#6F4F28")
                pareto_fig.add_scatter(
                    x=pareto_df["category"],
                    y=pareto_df["cum_share"],
                    name="Cumulative share",
                    yaxis="y2",
                    mode="lines+markers",
                    marker_color="#1F4E5F",
                )
                pareto_fig.update_layout(
                    template="plotly_white",
                    yaxis_title="Delay days",
                    yaxis2=dict(
                        overlaying="y",
                        side="right",
                        range=[0, 1],
                        showgrid=False,
                        tickformat=".0%",
                    ),
                    margin=dict(t=30, b=100),
                )
                selected_category = None
                if pareto_click and pareto_click.get("points"):
                    selected_category = pareto_click["points"][0]["x"]
                if selected_category:
                    category_mask = {selected_category}
                else:
                    category_mask = set(pareto_df.head(5)["category"].tolist())
                relevant_actions = []
                for _, step_row in steps_df.iterrows():
                    categories = json.loads(step_row.get("delay_days_by_category", "{}"))
                    if any(cat in category_mask for cat in categories):
                        relevant_actions.append(step_row["nepa_id"])
                relevant_actions = set(relevant_actions) if relevant_actions else set(steps_df["nepa_id"].unique())
                affected = (
                    steps_df[steps_df["nepa_id"].isin(relevant_actions)]
                    .assign(delay_days=lambda df: df["delay_days_total"])
                    .groupby("nepa_id", as_index=False)["delay_days"].sum()
                    .sort_values("delay_days", ascending=False)
                )
                affected_data = (
                    affected.head(10)[["nepa_id", "delay_days"]]
                    .merge(summary_df[["nepa_id", "project_id"]], on="nepa_id", how="left")
                    .rename(columns={"delay_days": "delay_days"})
                    .to_dict("records")
                )
                total_delay = delay_df["days"].sum()
                top_categories = pareto_df.head(3)
                bullets = []
                if total_delay > 0:
                    for _, row in top_categories.iterrows():
                        share = row["days"] / total_delay if total_delay else 0
                        bullets.append(
                            html.Li(
                                f"{row['category']}: {int(round(row['days'])):,} days ({share:.0%})."
                            )
                        )
                if selected_category and selected_category in pareto_df["category"].values:
                    focus_days = (
                        pareto_df.loc[
                            pareto_df["category"] == selected_category, "days"
                        ].iloc[0]
                    )
                    bullets.insert(
                        0,
                        html.Li(
                            f"Focused on {selected_category} covering {int(round(focus_days)):,} delay-days.",
                        ),
                    )
                pareto_note = html.Div(
                    [
                        html.P(
                            [
                                html.Span(
                                    f"{int(round(total_delay)):,} delay-days",
                                    className="insight-highlight",
                                ),
                                " captured after overlap adjustments.",
                            ],
                            className="insight-lead",
                        ),
                        html.Ul(bullets, className="insight-list") if bullets else None,
                    ],
                    className="insight-body",
                )

        # Agency benchmarking heatmap
        if summary_df.empty:
            agency_fig = empty_fig
            coop_fig = empty_fig
            litigation_fig = empty_fig
        else:
            benchmark = (
                summary_df.groupby(["lead_agency_id", "state"], dropna=False)["total_days"].median().unstack("state")
            )
            agency_fig = px.imshow(
                benchmark,
                color_continuous_scale="YlGnBu",
                aspect="auto",
                labels=dict(color="Median days"),
            )
            agency_fig.update_layout(template="plotly_white", margin=dict(t=40, b=40))

            coop_fig = px.scatter(
                summary_df,
                x="cooperating_agencies_count",
                y="total_days",
                color="action_level",
                trendline=None,
                labels={"cooperating_agencies_count": "Cooperating agencies", "total_days": "Total days"},
                color_discrete_sequence=["#4C6B75", "#C49A6C"],
            )
            if not summary_df["cooperating_agencies_count"].dropna().empty:
                x = summary_df["cooperating_agencies_count"].fillna(0).to_numpy()
                y = summary_df["total_days"].fillna(0).to_numpy()
                if len(x) > 1:
                    slope, intercept = np.polyfit(x, y, 1)
                    x_line = np.linspace(x.min(), x.max(), 100)
                    y_line = slope * x_line + intercept
                    coop_fig.add_traces(
                        go.Scatter(x=x_line, y=y_line, mode="lines", name="Trend", line=dict(color="#1F4E5F"))
                    )
                    if np.std(x) > 0 and np.std(y) > 0:
                        rho = np.corrcoef(x, y)[0, 1]
                        coop_fig.update_layout(
                            annotations=[
                                dict(
                                    text=f"ρ={rho:.2f} (N={len(x)})",
                                    xref="paper",
                                    yref="paper",
                                    x=0.95,
                                    y=0.05,
                                    showarrow=False,
                                )
                            ]
                        )
            coop_fig.update_layout(template="plotly_white")

            if permits_df.empty:
                litigation_fig = empty_fig
            else:
                permit_rates = (
                    permits_df.groupby("permit_type")
                    .agg(count=("permit_id", "count"), litigation_rate=("litigation_flag", "mean"))
                    .reset_index()
                )
                permit_rates["litigation_rate"] *= 100
                litigation_fig = px.bar(
                    permit_rates,
                    x="permit_type",
                    y="litigation_rate",
                    hover_data=["count"],
                    labels={"litigation_rate": "Litigation rate (%)", "permit_type": "Permit type"},
                    color="litigation_rate",
                    color_continuous_scale="Reds",
                )
                litigation_fig.update_layout(template="plotly_white", margin=dict(t=40, b=80))
                total_counts = permit_rates["count"].sum()
                weighted_rate = (
                    (permit_rates["litigation_rate"] * permit_rates["count"]).sum() / total_counts
                    if total_counts
                    else float("nan")
                )
                if not permit_rates.empty:
                    top_permit = permit_rates.sort_values("litigation_rate", ascending=False).iloc[0]
                    bullets = [
                        html.Li(
                            f"{top_permit['permit_type']} faces {top_permit['litigation_rate']:.1f}% litigation (n={top_permit['count']})."
                        )
                    ]
                    if len(permit_rates) > 1:
                        bottom_permit = permit_rates.sort_values("litigation_rate", ascending=True).iloc[0]
                        bullets.append(
                            html.Li(
                                f"Lowest exposure: {bottom_permit['permit_type']} at {bottom_permit['litigation_rate']:.1f}% (n={bottom_permit['count']})."
                            )
                        )
                    lead_text = (
                        [
                            html.Span(f"{weighted_rate:.1f}%", className="insight-highlight"),
                            " weighted litigation rate across permits.",
                        ]
                        if not np.isnan(weighted_rate)
                        else ["Permit litigation exposure varies by type."]
                    )
                    litigation_annotation = html.Div(
                        [
                            html.P(lead_text, className="insight-lead"),
                            html.Ul(bullets, className="insight-list"),
                        ],
                        className="insight-body",
                    )

        # Scenario metrics
        if scenario_steps.empty or scenario_summary.empty:
            copper_fig = empty_fig
            copper_annotation = html.Div("Select a scenario to see copper impacts.", className="insight-body")
            distribution_fig = empty_fig
            movers_data = []
            assumptions = "Select a scenario to view impacts."
        else:
            copper_fig, copper_annotation, movers_data = _build_copper_outputs(
                scenario_summary, projects_df, mines_df
            )
            distribution_fig = _build_distribution_figure(summary_df, scenario_summary)
            assumptions = _render_assumptions(params)
            copper_annotation = html.Div(copper_annotation, className="insight-body")

        story_bullets = []
        if top_step is not None and top_step_days is not None and not np.isnan(top_step_days):
            story_bullets.append(
                html.Li(
                    f"Step {top_step} alone absorbs {int(round(top_step_days)):,} median days.",
                )
            )
        if not np.isnan(litigation_pct):
            story_bullets.append(
                html.Li(f"Litigation touches {litigation_pct:.1f}% of actions in view."),
            )
        if not np.isnan(cap_pct):
            story_bullets.append(
                html.Li(f"Cap compliance: {cap_pct:.1f}% meeting NOI→ROD/EA limits."),
            )
        if n_actions < 10:
            story_bullets.append(html.Li("Sample <10 actions — interpret cautiously."))
        if not np.isnan(median_total):
            lead_children = [
                html.Span(f"{int(round(median_total)):,} days", className="insight-highlight"),
                " median NEPA window",
            ]
            if not np.isnan(median_delay) and median_delay > 0:
                lead_children.extend(
                    [
                        "; ",
                        html.Span(f"{int(round(median_delay)):,} days", className="insight-highlight"),
                        " of in-window delay.",
                    ]
                )
            else:
                lead_children.append(" with limited recorded delay.")
            timeline_children = [html.P(lead_children, className="insight-lead")]
            if story_bullets:
                timeline_children.append(html.Ul(story_bullets, className="insight-list"))
            timeline_story = html.Div(timeline_children, className="insight-body")

        qa_data = qa_df.to_dict("records")

        return (
            kpi_total,
            kpi_delay,
            kpi_lit,
            kpi_cap,
            kpi_total_ctx,
            kpi_delay_ctx,
            kpi_lit_ctx,
            kpi_cap_ctx,
            timeline_story,
            waterfall_fig,
            waterfall_note,
            variance_fig,
            variance_note,
            step_summary,
            pareto_fig,
            pareto_note,
            affected_data,
            agency_fig,
            coop_fig,
            litigation_fig,
            litigation_annotation,
            copper_fig,
            copper_annotation,
            distribution_fig,
            movers_data,
            assumptions,
            qa_data,
        )


def _build_copper_outputs(summary_df, projects_df, mines_df):
    if summary_df.empty:
        return go.Figure(), "", []
    merged = summary_df.merge(projects_df, on="project_id", how="left", suffixes=("", "_project"))
    merged = merged.merge(mines_df, on="mine_id", how="left", suffixes=("", "_mine"))
    merged["nameplate_cu_ktpa"] = merged["nameplate_cu_ktpa"].fillna(0)
    merged["lifecycle_years"] = merged["lifecycle_years"].fillna(20)
    merged["annual_production"] = merged["nameplate_cu_ktpa"].clip(lower=0)
    merged["deferred_copper"] = (
        merged["scenario_delay_days_total"].fillna(0) / 365.0
    ) * merged["annual_production"]
    merged["scenario_production"] = (merged["annual_production"] - merged["deferred_copper"]).clip(lower=0)
    merged_sorted = merged.sort_values("deferred_copper", ascending=False).reset_index(drop=True)
    merged_sorted["rank"] = merged_sorted.index + 1

    plot_df = pd.DataFrame(
        {
            "rank": merged_sorted["rank"],
            "Baseline": merged_sorted["annual_production"].cumsum(),
            "Scenario": merged_sorted["scenario_production"].cumsum(),
        }
    )
    copper_fig = go.Figure()
    copper_fig.add_trace(
        go.Scatter(x=plot_df["rank"], y=plot_df["Baseline"], mode="lines", name="Baseline", line=dict(color="#4C6B75"))
    )
    copper_fig.add_trace(
        go.Scatter(x=plot_df["rank"], y=plot_df["Scenario"], mode="lines", name="Scenario", line=dict(color="#C49A6C"))
    )
    copper_fig.update_layout(
        template="plotly_white",
        xaxis_title="Cumulative actions",
        yaxis_title="Cumulative production (kt)",
        legend_orientation="h",
        legend_y=-0.2,
    )
    median_deferred = merged["deferred_copper"].median()
    total_deferred = merged["deferred_copper"].sum()
    top_project = merged_sorted.iloc[0] if not merged_sorted.empty else None
    highlight = (
        f"Top mover: {top_project['nepa_id']} with {top_project['deferred_copper']:.1f} kt deferred."
        if top_project is not None
        else ""
    )
    copper_annotation = (
        f"P50 deferred copper {median_deferred:.1f} kt; portfolio impact {total_deferred:.1f} kt across {len(merged)} actions. {highlight}".strip()
    )
    if not copper_annotation.endswith("."):
        copper_annotation += "."
    movers = (
        merged[["nepa_id", "total_days", "scenario_total_days"]]
        .assign(days_saved=lambda df: df["total_days"] - df["scenario_total_days"])
        .sort_values("days_saved", ascending=False)
        .head(10)
    )
    movers["scenario"] = "Scenario"
    movers_data = movers[["nepa_id", "days_saved", "scenario"]].to_dict("records")
    return copper_fig, copper_annotation, movers_data


def _build_distribution_figure(baseline_summary, scenario_summary):
    if baseline_summary.empty or scenario_summary.empty:
        return go.Figure()
    baseline = baseline_summary[["nepa_id", "total_days"]].assign(scenario="Baseline")
    scenario = scenario_summary[["nepa_id", "scenario_total_days"]].rename(
        columns={"scenario_total_days": "total_days"}
    )
    scenario["scenario"] = "Scenario"
    combined = pd.concat([baseline, scenario], ignore_index=True)
    fig = px.violin(
        combined,
        x="scenario",
        y="total_days",
        color="scenario",
        box=True,
        points="outliers",
        color_discrete_sequence=["#4C6B75", "#C49A6C"],
    )
    fig.update_layout(template="plotly_white", showlegend=False, yaxis_title="Total days")
    return fig


def _render_assumptions(params: Dict[str, object]) -> str:
    if not params:
        return ""
    lines = [f"Scenario: {params.get('scenario', 'baseline').title()}"]
    if params.get("scenario") == "trump2020":
        lines.append("• Actions forced to meet NOI→ROD cap; steps B & D compressed proportionally.")
    if params.get("scenario") == "speed":
        lines.append("• Modeling rework & policy-change delays scaled by user settings.")
        lines.append("• Litigation days scaled by slider; reuse scale applied to compressible steps.")
    lines.append(f"• EIS cap: {params.get('eis_cap_days', 730)} days; EA cap: {params.get('ea_cap_days', 365)} days.")
    lines.append(
        f"• Litigation scale={params.get('litigation_scale', 1.0):.2f}, Rework scale={params.get('rework_scale', 1.0):.2f}, "
        f"Policy scale={params.get('policy_change_scale', 1.0):.2f}, Reuse scale={params.get('reuse_scale', 1.0):.2f}."
    )
    return html.Ul([html.Li(line) for line in lines])
