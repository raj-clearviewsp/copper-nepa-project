"""Dash application entry point for the copper NEPA dashboard."""

from __future__ import annotations

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from src.data.ingest import LoadedData, load_workbook
from src.data.transformations import derive_tables
from src.dashboards.callbacks import register_callbacks
from src.dashboards.layout import create_layout
from src.utils.config import DEFAULT_CONFIG

DATA_PATH = Path("updated-permitting-data-template.xlsx")


def initialize_data(data_path: Path) -> dict:
    loaded: LoadedData = load_workbook(data_path)
    action_steps, action_summary, qa = derive_tables(loaded, DEFAULT_CONFIG)
    data_context = {
        "action_steps": action_steps,
        "action_summary": action_summary,
        "qa": qa,
        "delays": loaded.delays,
        "projects": loaded.projects,
        "mines": loaded.mines,
        "permits": loaded.permits,
        "comments": loaded.comments,
    }
    return data_context


def create_app() -> dash.Dash:
    data_context = initialize_data(DATA_PATH)
    external_stylesheets = [dbc.themes.MATERIA]
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
    app.title = "Copper NEPA Dashboard"
    app.layout = create_layout(app, data_context)
    register_callbacks(app, DEFAULT_CONFIG)
    return app


app = create_app()
server = app.server


if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=8050)
