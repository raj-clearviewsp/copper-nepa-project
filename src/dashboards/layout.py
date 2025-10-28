"""Dash layout definitions for the NEPA copper dashboard."""

from __future__ import annotations

from typing import Dict, List

import dash_bootstrap_components as dbc
from dash import dcc, html

from src.data.ingest import to_json_records


FILTER_STYLE = {"marginBottom": "1rem"}


def _build_filter_dropdown(id_, label, options, multi=True):
    return dbc.FormGroup(
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
            dbc.CardHeader("Global Filters"),
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
                    dbc.FormGroup(
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
                    dbc.FormGroup(
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
                ]
            ),
        ],
        className="filter-panel",
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
    return dbc.Container(
        [
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
                        dbc.Card(
                            [
                                dbc.CardHeader("Median NEPA waterfall (A–E)"),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(id="fig-waterfall", config={"displaylogo": False}),
                                        html.Div(id="waterfall-annotation", className="chart-annotation"),
                                    ]
                                ),
                            ]
                        ),
                        md=8,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Step variance"),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(id="fig-variance", config={"displaylogo": False}),
                                        html.Div(id="variance-annotation", className="chart-annotation"),
                                    ]
                                ),
                            ]
                        ),
                        md=4,
                    ),
                ],
                className="g-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Step summary"),
                                dbc.CardBody(
                                    dcc.Loading(
                                        dcc.DataTable(
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
                                    ),
                                ),
                            ]
                        ),
                    ),
                ],
                className="g-3",
            ),
        ],
        fluid=True,
        className="tab-content",
    )


def _build_drivers_tab():
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Delay Pareto"),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(id="fig-delay-pareto", config={"displaylogo": False}),
                                        html.Div(id="pareto-annotation", className="chart-annotation"),
                                    ]
                                ),
                            ]
                        ),
                        md=7,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Affected actions"),
                                dbc.CardBody(
                                    dcc.DataTable(
                                        id="table-affected-actions",
                                        columns=[
                                            {"name": "NEPA ID", "id": "nepa_id"},
                                            {"name": "Project", "id": "project_id"},
                                            {"name": "Delay days", "id": "delay_days"},
                                        ],
                                        style_table={"overflowY": "auto", "maxHeight": "400px"},
                                    ),
                                ),
                            ]
                        ),
                        md=5,
                    ),
                ],
                className="g-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Agency × State benchmarking"),
                                dbc.CardBody(dcc.Graph(id="fig-agency-benchmark", config={"displaylogo": False})),
                            ]
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Cooperation vs timeline"),
                                dbc.CardBody(dcc.Graph(id="fig-cooperation", config={"displaylogo": False})),
                            ]
                        ),
                        md=6,
                    ),
                ],
                className="g-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Litigation by permit type"),
                                dbc.CardBody(dcc.Graph(id="fig-litigation", config={"displaylogo": False})),
                            ]
                        ),
                    )
                ],
                className="g-3",
            ),
        ],
        fluid=True,
        className="tab-content",
    )


def _build_scenarios_tab():
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Scenario controls"),
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
                            ]
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Copper at risk"),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(id="fig-copper", config={"displaylogo": False}),
                                        html.Div(id="copper-summary", className="chart-annotation"),
                                    ]
                                ),
                            ]
                        ),
                        md=8,
                    ),
                ],
                className="g-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Before/After distributions"),
                                dbc.CardBody(dcc.Graph(id="fig-distribution", config={"displaylogo": False})),
                            ]
                        ),
                    ),
                ],
                className="g-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Top movers"),
                                dbc.CardBody(
                                    dcc.DataTable(
                                        id="table-top-movers",
                                        columns=[
                                            {"name": "NEPA ID", "id": "nepa_id"},
                                            {"name": "Days saved", "id": "days_saved"},
                                            {"name": "Scenario", "id": "scenario"},
                                        ],
                                    )
                                ),
                            ]
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Assumptions"),
                                dbc.CardBody(
                                    html.Div(id="scenario-assumptions", className="assumptions-panel"),
                                ),
                            ]
                        ),
                        md=6,
                    ),
                ],
                className="g-3",
            ),
        ],
        fluid=True,
        className="tab-content",
    )


def _build_qa_tab():
    return dbc.Container(
        [
            dbc.Card(
                [
                    dbc.CardHeader("Data QA flags"),
                    dbc.CardBody(
                        dcc.DataTable(
                            id="table-qa",
                            columns=[
                                {"name": "NEPA ID", "id": "nepa_id"},
                                {"name": "Issue", "id": "flag_reason"},
                            ],
                            style_table={"overflowY": "auto", "maxHeight": "500px"},
                        )
                    ),
                ]
            )
        ],
        fluid=True,
        className="tab-content",
    )


def create_layout(app, data_context: Dict[str, object]) -> html.Div:
    summary_df = data_context["action_summary"]
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
            html.H1("Copper NEPA Executive Dashboard", className="app-title"),
            html.P(
                "Interactive intelligence on NEPA actions for U.S. copper mines, built for policy leaders.",
                className="app-subtitle",
            ),
            dbc.Row(
                [
                    dbc.Col(_build_filter_panel(summary_df), md=3),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        dcc.Tabs(
                                            id="main-tabs",
                                            value="tab-timelines",
                                            children=[
                                                dcc.Tab(label="Timelines & predictability", value="tab-timelines"),
                                                dcc.Tab(label="Drivers & benchmarks", value="tab-drivers"),
                                                dcc.Tab(label="Impact & scenarios", value="tab-scenarios"),
                                                dcc.Tab(label="Data QA", value="tab-qa"),
                                            ],
                                        ),
                                        html.Div(id="tab-content"),
                                    ]
                                )
                            ]
                        ),
                        md=9,
                    ),
                ],
                className="g-3",
            ),
        ],
        fluid=True,
        className="app-container",
    )


TAB_CONTENT_MAP = {
    "tab-timelines": _build_timeline_tab(),
    "tab-drivers": _build_drivers_tab(),
    "tab-scenarios": _build_scenarios_tab(),
    "tab-qa": _build_qa_tab(),
}
