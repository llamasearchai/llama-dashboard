# src/llama_dashboard/app.py

import httpx  # For making API calls
import pandas as pd
import streamlit as st

from . import __version__

# Basic page configuration
st.set_page_config(
    page_title=f"Llama Dashboard v{__version__}",
    page_icon="🦙",
    layout="wide",
)

# --- API Client (Example) ---
# Replace with actual API endpoint from config/env var
LLAMA_API_BASE_URL = "http://localhost:8000/api/v1"  # Example: LlamaSearchAI API

# Use a session state for the client to avoid recreating it on every rerun
if "api_client" not in st.session_state:
    st.session_state.api_client = httpx.Client(base_url=LLAMA_API_BASE_URL, timeout=10.0)

api_client = st.session_state.api_client


# --- Helper Functions (Example) ---
@st.cache_data(ttl=60)  # Cache for 1 minute
def get_api_status():
    """Fetches the status from the backend API."""
    try:
        # Adjust endpoint based on actual API (root or /health)
        response = api_client.get("/")
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except httpx.RequestError as e:
        st.error(f"Error connecting to API: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None


# --- Sidebar ---
st.sidebar.title(f"Llama Dashboard v{__version__}")
# Add navigation or controls to the sidebar
page = st.sidebar.radio("Navigate", ["Overview", "Search", "Monitoring"], key="navigation")

st.sidebar.markdown("--- ")

# Display API Status in Sidebar
st.sidebar.subheader("API Status")
api_status_data = get_api_status()
if api_status_data:
    st.sidebar.json(api_status_data)
    st.sidebar.success("API Connected")
else:
    st.sidebar.error("API Connection Failed")

# --- Main Page Content ---
st.title(f"🦙 Llama Dashboard - {page}")

if page == "Overview":
    st.header("System Overview")
    st.markdown("Welcome to the LlamaAI Ecosystem Dashboard.")
    st.info("This section will display overall system health and key metrics.")
    # Add placeholder charts or summaries
    chart_data = pd.DataFrame(
        [[10, 20, 30], [15, 25, 35], [12, 22, 32]],
        columns=["Service A", "Service B", "Service C"],
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    st.line_chart(chart_data)

elif page == "Search":
    st.header("Interactive Search")
    st.info("This section will provide an interface to query the search API.")

    query = st.text_input("Enter your search query:", key="search_query")
    if query:
        st.write(f"Searching for: '{query}'")
        # --- Add API call logic here ---
        # try:
        #     params = {"query": query, "limit": 10}
        #     response = api_client.get("/search", params=params) # Adjust endpoint
        #     response.raise_for_status()
        #     results = response.json()
        #     st.success(f"Found {len(results.get('results', []))} results:")
        #     st.json(results) # Display raw results for now
        #     # Or display in a more structured way (e.g., DataFrame)
        #     # if results.get('results'):
        #     #     st.dataframe(pd.DataFrame(results['results']))
        # except httpx.RequestError as e:
        #     st.error(f"API request failed: {e}")
        # except Exception as e:
        #     st.error(f"Search failed: {e}")
        st.warning("Search API call placeholder - implement actual call.")

elif page == "Monitoring":
    st.header("Service Monitoring")
    st.info("This section will display detailed monitoring data for individual services.")
    # Placeholder for monitoring data
    st.warning("Monitoring display placeholder - fetch and display data.")


# --- Entry point for script execution ---
def run():
    """Function called by the script entry point in pyproject.toml"""
    # Streamlit apps are typically run via `streamlit run app.py`
    # This function might just print instructions or try to launch streamlit programmatically (less common)
    print("To run the dashboard, execute:")
    print("streamlit run src/llama_dashboard/app.py")


if __name__ == "__main__":
    # This block allows running `python src/llama_dashboard/app.py`
    # but the standard way is `streamlit run src/llama_dashboard/app.py`
    # Running this way might not behave exactly like the streamlit command
    st.warning("Running via __main__; prefer `streamlit run src/llama_dashboard/app.py`")
    # The Streamlit elements above will execute when this script is imported or run.
    pass  # No explicit run call needed here for Streamlit's execution model
