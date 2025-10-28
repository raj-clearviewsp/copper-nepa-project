"""Dash layout definitions for the NEPA copper dashboard."""

from __future__ import annotations

from typing import Dict, List, Optional

import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table

from src.data.ingest import to_json_records


FILTER_STYLE = {"marginBottom": "1rem"}


def _section_header(title: str, subtitle: Optional[str] = None) -> html.Div:
    children: List = [html.H3(title, className="section-title")]
    if subtitle:
        children.append(html.P(subtitle, className="section-subtitle"))
    return html.Div(children, className="section-header")


def _chart_card(card_title: str, figure_id: str) -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardHeader(card_title, className="card-heading"),
            dbc.CardBody(
                dcc.Graph(id=figure_id, config={"displaylogo": False}),
                className="card-body-chart",
            ),
        ],
        className="story-card chart-card",
    )


def _insight_card(title: str, body_id: str, subtitle: Optional[str] = None) -> dbc.Card:
    body_children: List = []
    if subtitle:
        body_children.append(html.P(subtitle, className="insight-subtitle"))
    body_children.append(html.Div(id=body_id, className="insight-body"))
    return dbc.Card(
        [
            dbc.CardHeader(title, className="card-heading"),
            dbc.CardBody(body_children, className="insight-card-body"),
        ],
        className="story-card insight-card",
    )


def _table_card(title: str, table_component: dash_table.DataTable) -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardHeader(title, className="card-heading"),
            dbc.CardBody(dcc.Loading(table_component, className="table-loading")),
        ],
        className="story-card table-card",
    )


def _build_filter_dropdown(id_, label, options, multi=True):
    return html.Div(
        [
            dbc.Label(label, className="filter-label"),
            dcc.Dropdown(
                id=id_,
                options=[{"label": opt, "value": opt} for opt in options if opt or opt == 0],
                multi=multi,
                placeholder=f"Select {label.lower()}",
            ),
        ],
        style=FILTER_STYLE,
    )


def _build_filter_panel(summary_df) -> dbc.Card:
    return dbc.Card(
        [
            dbc.CardHeader("Global filters", className="card-heading"),
            dbc.CardBody(
                [
                    _build_filter_dropdown(
                        "filter-action-level", "Action Level", sorted(summary_df["action_level"].dropna().unique()), multi=True
                    ),
                    _build_filter_dropdown("filter-state", "State", sorted(summary_df["state"].dropna().unique())),
                    _build_filter_dropdown(
                        "filter-lead-agency",
                        "Lead Agency",
                        sorted(summary_df["lead_agency_id"].dropna().unique()),
                    ),
                    _build_filter_dropdown(
                        "filter-cohort-year",
                        "Cohort Year",
                        sorted(summary_df["cohort_year"].dropna().unique()),
                    ),
                    _build_filter_dropdown(
                        "filter-cohort-decade",
                        "Cohort Decade",
                        sorted(summary_df["cohort_decade"].dropna().unique()),
                    ),
                    _build_filter_dropdown(
                        "filter-mine-type", "Mine Type", sorted(summary_df["mine_type"].dropna().unique())
                    ),
                    _build_filter_dropdown(
                        "filter-tailings-type",
                        "Tailings Type",
                        sorted(summary_df["tailings_type"].dropna().unique()),
                    ),
                    html.Div(
                        [
                            dbc.Label("Litigation"),
                            dbc.RadioItems(
                                id="filter-litigation",
                                options=[
                                    {"label": "All", "value": "all"},
                                    {"label": "With litigation", "value": "yes"},
                                    {"label": "No litigation", "value": "no"},
                                ],
                                value="all",
                                inline=True,
                            ),
                        ],
                        style=FILTER_STYLE,
                    ),
                    html.Div(
                        [
                            dbc.Label("Cooperating Agencies"),
                            dcc.RangeSlider(
                                id="filter-coop-count",
                                min=0,
                                max=int(summary_df["cooperating_agencies_count"].fillna(0).max() or 0),
                                step=1,
                                value=[
                                    0,
                                    int(summary_df["cooperating_agencies_count"].fillna(0).max() or 0),
                                ],
                                tooltip={"placement": "bottom"},
                            ),
                        ],
                        style=FILTER_STYLE,
                    ),
                    dbc.Checklist(
                        options=[
                            {"label": "Include flagged actions", "value": "include_flagged"},
                            {"label": "Include active/incomplete actions", "value": "include_active"},
                        ],
                        id="filter-flags",
                        value=[],
                        switch=True,
                    ),
                ],
                className="filter-body",
            ),
        ],
        className="filter-card",
    )


def _kpi_card(id_, title):
    return dbc.Col(
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H6(title, className="kpi-title"),
                        html.H2(id=id_, className="kpi-value"),
                        html.Div(id=f"{id_}-context", className="kpi-context"),
                    ]
                )
            ],
            className="kpi-card",
        ),
        md=3,
        sm=6,
        xs=12,
    )


def _build_timeline_tab():
    step_summary_table = dash_table.DataTable(
        id="table-step-summary",
        columns=[
            {"name": "Step", "id": "step"},
            {"name": "Median days", "id": "median_days"},
            {"name": "P10", "id": "p10"},
            {"name": "P90", "id": "p90"},
            {"name": "N", "id": "N"},
        ],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center"},
    )

    return dbc.Container(
        [
            _section_header(
                "Timelines & predictability",
                "Quantify the NEPA journey and spotlight where copper actions lose time.",
            ),
            dbc.Row(
                [
                    _kpi_card("kpi-total-days", "Median NEPA days"),
                    _kpi_card("kpi-delay-days", "Median in-window delay days"),
                    _kpi_card("kpi-litigation", "% with litigation"),
                    _kpi_card("kpi-cap", "% meeting statutory cap"),
                ],
                className="g-3 kpi-row",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        _insight_card(
                            "Executive brief",
                            "timeline-story",
                            subtitle="What the current filters say about time, delays, and predictability.",
                        ),
                        md=12,
                    )
                ],
                className="g-3 story-row",
            ),
            dbc.Row(
                [
                    dbc.Col(_chart_card("Median timeline blueprint", "fig-waterfall"), md=8),
                    dbc.Col(
                        _insight_card(
                            "Where the clock stretches",
                            "waterfall-annotation",
                            subtitle="Median step lengths and the contribution of in-window delays.",
                        ),
                        md=4,
                    ),
                ],
                className="g-3 story-row",
                align="stretch",
            ),
            dbc.Row(
                [
                    dbc.Col(_chart_card("Predictability profile", "fig-variance"), md=7),
                    dbc.Col(
                        [
                            _insight_card(
                                "Volatility watch",
                                "variance-annotation",
                                subtitle="Steps with the widest swing between fast and slow actions.",
                            ),
                            _table_card("Step reference", step_summary_table),
                        ],
                        md=5,
                    ),
                ],
                className="g-3 story-row",
                align="stretch",
            ),
        ],
        fluid=True,
        className="tab-wrapper",
    )


def _build_drivers_tab():
    affected_table = dash_table.DataTable(
        id="table-affected-actions",
        columns=[
            {"name": "NEPA ID", "id": "nepa_id"},
            {"name": "Project", "id": "project_id"},
            {"name": "Delay days", "id": "delay_days"},
        ],
        style_table={"overflowY": "auto", "maxHeight": "360px"},
        style_cell={"textAlign": "left"},
    )

    return dbc.Container(
        [
            _section_header(
                "Drivers & benchmarks",
                "Diagnose structural bottlenecks and benchmark performance across agencies and jurisdictions.",
            ),
            dbc.Row(
                [
                    dbc.Col(_chart_card("Delay Pareto", "fig-delay-pareto"), md=8),
                    dbc.Col(
                        [
                            _insight_card(
                                "Delay storyline",
                                "pareto-annotation",
                                subtitle="Top categories where days accumulate after overlap adjustments.",
                            ),
                            _table_card("Priority actions", affected_table),
                        ],
                        md=4,
                    ),
                ],
                className="g-3 story-row",
                align="stretch",
            ),
            dbc.Row(
                [
                    dbc.Col(_chart_card("Agency × State benchmarking", "fig-agency-benchmark"), md=6),
                    dbc.Col(_chart_card("Cooperation vs total days", "fig-cooperation"), md=6),
                ],
                className="g-3 story-row",
                align="stretch",
            ),
            dbc.Row(
                [
                    dbc.Col(_chart_card("Litigation by permit type", "fig-litigation"), md=8),
                    dbc.Col(
                        _insight_card(
                            "Litigation signal",
                            "litigation-annotation",
                            subtitle="Which permit pathways are most exposed to legal challenge.",
                        ),
                        md=4,
                    ),
                ],
                className="g-3 story-row",
                align="stretch",
            ),
        ],
        fluid=True,
        className="tab-wrapper",
    )


def _build_scenarios_tab():
    movers_table = dash_table.DataTable(
        id="table-top-movers",
        columns=[
            {"name": "NEPA ID", "id": "nepa_id"},
            {"name": "Days saved", "id": "days_saved"},
            {"name": "Scenario", "id": "scenario"},
        ],
        style_table={"overflowY": "auto", "maxHeight": "320px"},
        style_cell={"textAlign": "left"},
    )

    control_card = dbc.Card(
        [
            dbc.CardHeader("Scenario controls", className="card-heading"),
            dbc.CardBody(
                [
                    dbc.RadioItems(
                        id="scenario-selector",
                        options=[
                            {"label": "Baseline", "value": "baseline"},
                            {"label": "Trump 2020 cap", "value": "trump2020"},
                            {"label": "SPEED Act", "value": "speed"},
                        ],
                        value="baseline",
                        inline=True,
                        className="scenario-selector",
                    ),
                    html.Hr(),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("EIS cap (days)"),
                                    dcc.Input(id="input-eis-cap", type="number", min=1, value=730),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("EA cap (days)"),
                                    dcc.Input(id="input-ea-cap", type="number", min=1, value=365),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Reuse scale"),
                                    dcc.Slider(
                                        id="input-reuse-scale",
                                        min=0.2,
                                        max=1.0,
                                        step=0.05,
                                        value=1.0,
                                        marks=None,
                                        tooltip={"placement": "bottom", "always_visible": False},
                                    ),
                                ],
                                md=4,
                            ),
                        ],
                        className="g-2",
                    ),
                    html.Br(),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Litigation scale"),
                                    dcc.Slider(
                                        id="input-litigation-scale",
                                        min=0.0,
                                        max=1.0,
                                        step=0.05,
                                        value=1.0,
                                        tooltip={"placement": "bottom"},
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Modeling rework scale"),
                                    dcc.Slider(
                                        id="input-rework-scale",
                                        min=0.0,
                                        max=1.0,
                                        step=0.05,
                                        value=1.0,
                                        tooltip={"placement": "bottom"},
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Policy change scale"),
                                    dcc.Slider(
                                        id="input-policy-scale",
                                        min=0.0,
                                        max=1.0,
                                        step=0.05,
                                        value=1.0,
                                        tooltip={"placement": "bottom"},
                                    ),
                                ],
                                md=4,
                            ),
                        ],
                        className="g-2",
                    ),
                ]
            ),
        ],
        className="story-card control-card",
    )

    return dbc.Container(
        [
            _section_header(
                "Impact & scenarios",
                "Stress-test statutory levers to see how copper production and decision timelines respond.",
            ),
            dbc.Row(
                [
                    dbc.Col(control_card, md=4),
                    dbc.Col(
                        [
                            _chart_card("Copper at risk", "fig-copper"),
                            _insight_card(
                                "Production headline",
                                "copper-summary",
                                subtitle="Median deferred copper under the selected scenario versus baseline.",
                            ),
                        ],
                        md=8,
                    ),
                ],
                className="g-3 story-row",
                align="stretch",
            ),
            dbc.Row(
                [
                    dbc.Col(_chart_card("Before vs after distribution", "fig-distribution"), md=7),
                    dbc.Col(_table_card("Top movers", movers_table), md=5),
                ],
                className="g-3 story-row",
                align="stretch",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        _insight_card(
                            "Scenario assumptions",
                            "scenario-assumptions",
                            subtitle="Reference the policy levers and compression factors applied.",
                        ),
                        md=12,
                    )
                ],
                className="g-3 story-row",
            ),
        ],
        fluid=True,
        className="tab-wrapper",
    )


def _build_qa_tab():
    qa_table = dash_table.DataTable(
        id="table-qa",
        columns=[
            {"name": "NEPA ID", "id": "nepa_id"},
            {"name": "Issue", "id": "flag_reason"},
        ],
        style_table={"overflowY": "auto", "maxHeight": "520px"},
        style_cell={"textAlign": "left"},
    )

    return dbc.Container(
        [
            _section_header(
                "Data QA",
                "Diagnostics on incomplete milestones or date inconsistencies surfaced during ingest.",
            ),
            dbc.Row(
                [
                    dbc.Col(_table_card("Flagged records", qa_table), md=12),
                ],
                className="g-3 story-row",
            ),
        ],
        fluid=True,
        className="tab-wrapper",
    )


def create_layout(app, data_context: Dict[str, object]) -> html.Div:
    summary_df = data_context["action_summary"]
    hero_section = html.Div(
        [
            html.Div(
                [
                    html.H1("Copper NEPA Executive Dashboard", className="app-title"),
                    html.P(
                        "A policy-ready view of how U.S. copper mine actions move through NEPA, where days stack up, and how reforms shift production.",
                        className="app-subtitle",
                    ),
                    html.Div(
                        [
                            html.Div("Timelines", className="hero-pill"),
                            html.Div("Drivers", className="hero-pill"),
                            html.Div("Scenarios", className="hero-pill"),
                        ],
                        className="hero-pills",
                    ),
                ],
                className="hero-copy",
            )
        ],
        className="hero-section",
    )
    return dbc.Container(
        [
            dcc.Store(id="store-action-steps", data=to_json_records(data_context["action_steps"])),
            dcc.Store(id="store-action-summary", data=to_json_records(summary_df)),
            dcc.Store(id="store-delays", data=to_json_records(data_context["delays"])),
            dcc.Store(id="store-projects", data=to_json_records(data_context["projects"])),
            dcc.Store(id="store-permits", data=to_json_records(data_context["permits"])),
            dcc.Store(id="store-mines", data=to_json_records(data_context["mines"])),
            dcc.Store(id="store-comments", data=to_json_records(data_context["comments"])),
            dcc.Store(id="store-qa", data=to_json_records(data_context["qa"])),
            dcc.Store(id="store-filtered", data="{}"),
            dcc.Store(id="store-scenario", data="{}"),
            hero_section,
            dbc.Row(
                [
                    dbc.Col(_build_filter_panel(summary_df), md=3, className="filter-column"),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        dcc.Tabs(
                                            id="main-tabs",
                                            value="tab-timelines",
                                            children=[
                                                dcc.Tab(
                                                    label="Timelines & predictability",
                                                    value="tab-timelines",
                                                    children=_build_timeline_tab(),
                                                ),
                                                dcc.Tab(
                                                    label="Drivers & benchmarks",
                                                    value="tab-drivers",
                                                    children=_build_drivers_tab(),
                                                ),
                                                dcc.Tab(
                                                    label="Impact & scenarios",
                                                    value="tab-scenarios",
                                                    children=_build_scenarios_tab(),
                                                ),
                                                dcc.Tab(
                                                    label="Data QA",
                                                    value="tab-qa",
                                                    children=_build_qa_tab(),
                                                ),
                                            ],
                                        ),
                                    ],
                                    className="tab-card-body",
                                )
                            ]
                        ),
                        md=9,
                        className="tabs-column",
                    ),
                ],
                className="g-4 main-content-row",
            ),
        ],
        fluid=True,
        className="app-container",
    )
