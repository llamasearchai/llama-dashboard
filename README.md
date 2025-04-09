# llama-dashboard

[![PyPI version](https://img.shields.io/pypi/v/llama_dashboard.svg)](https://pypi.org/project/llama_dashboard/)
[![License](https://img.shields.io/github/license/llamasearchai/llama-dashboard)](https://github.com/llamasearchai/llama-dashboard/blob/main/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/llama_dashboard.svg)](https://pypi.org/project/llama_dashboard/)
[![CI Status](https://github.com/llamasearchai/llama-dashboard/actions/workflows/llamasearchai_ci.yml/badge.svg)](https://github.com/llamasearchai/llama-dashboard/actions/workflows/llamasearchai_ci.yml)

**Llama Dashboard (llama-dashboard)** provides a web-based dashboard application for visualizing data and monitoring systems within the LlamaSearch AI ecosystem. It connects to various data sources to present information graphically.

## Key Features

- **Web Application:** A dashboard application, likely built with a framework like Flask or Streamlit (`app.py`).
- **Data Source Connectors:** Components to fetch data from different sources (databases, APIs, logs) (`data_sources.py`).
- **Visualization:** Displays data using charts, graphs, and tables.
- **Core Module:** Manages application setup and data fetching logic (`core.py`).
- **Configurable:** Allows defining data sources, visualization types, and refresh rates (`config.py`).

## Installation

```bash
pip install llama-dashboard
# Or install directly from GitHub for the latest version:
# pip install git+https://github.com/llamasearchai/llama-dashboard.git
```

## Usage

*(Instructions on how to run the dashboard application will be added here.)*

```bash
# Example command to run the dashboard
llama-dashboard run --config dashboard_config.yaml
# Then access via http://localhost:8050 (or similar)
```

## Architecture Overview

```mermaid
graph TD
    A[User (Web Browser)] --> B{Dashboard Web App (app.py)};
    B -- Requests Data --> C{Core Module (core.py)};
    C --> D{Data Source Interface (data_sources.py)};
    D -- Fetches Data --> E[(Data Source 1: DB)];
    D -- Fetches Data --> F[(Data Source 2: API)];
    D -- Fetches Data --> G[(Data Source 3: Logs)];
    E --> D;
    F --> D;
    G --> D;
    D --> C;
    C -- Formats Data --> B;
    B -- Renders UI / Visualizations --> A;

    H[Configuration (config.py)] -- Configures --> C;
    H -- Configures --> D;

    style B fill:#f9f,stroke:#333,stroke-width:2px
    style E fill:#ccf,stroke:#333,stroke-width:1px
    style F fill:#ccf,stroke:#333,stroke-width:1px
    style G fill:#ccf,stroke:#333,stroke-width:1px
```

1.  **User Interface:** The user accesses the dashboard through a web browser.
2.  **Web Application:** Handles user requests and renders the dashboard UI.
3.  **Core Module:** Orchestrates data fetching based on the dashboard configuration.
4.  **Data Source Interface:** Connects to and retrieves data from various configured backends.
5.  **Data Formatting:** Data is processed and formatted for visualization.
6.  **Rendering:** The web application displays the data using charts and other UI elements.
7.  **Configuration:** Defines data sources, connection details, visualization types, refresh intervals, etc.

## Configuration

*(Details on configuring data source connections (DB URIs, API endpoints/keys), dashboard layouts, chart types, etc., will be added here.)*

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/llamasearchai/llama-dashboard.git
cd llama-dashboard

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

### Testing

```bash
pytest tests/
```

### Contributing

Contributions are welcome! Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) and submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Developed by lalamasearhc.*
