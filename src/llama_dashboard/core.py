"""
llama_dashboard/dashboard_service.py
Main orchestration service for the dashboard system.
"""

import threading
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from llama_dashboard.ar_vr import ARVRIntegrationManager
from llama_dashboard.data_sources import BaseDataSource, DataSourceRegistry
from llama_dashboard.dp import DifferentialPrivacyEngine
from llama_dashboard.insights import InsightGenerator
from llama_dashboard.preprocessing import NeuralPreprocessor
from llama_dashboard.security import CredentialManager, Encryptor
from llama_dashboard.utils.exceptions import (
    DashboardConfigError,
    DataSourceError,
    VisualizationError,
)
from llama_dashboard.utils.logging import setup_logger
from llama_dashboard.visualization import MLXVisualizationBackend


class UpdateFrequency(Enum):
    """Update frequency options for dashboard refresh."""

    REALTIME = "realtime"  # Continuous updates
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"  # Custom interval in seconds


class DashboardState:
    """Thread-safe container for dashboard state."""

    def __init__(self):
        self._state = {}
        self._lock = threading.RLock()
        self._last_updated = datetime.now()

    def update(self, key: str, value: Any) -> None:
        """Update a state value safely.

        Args:
            key: State key to update
            value: New value to set
        """
        with self._lock:
            self._state[key] = value
            self._last_updated = datetime.now()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a state value safely.

        Args:
            key: State key to retrieve
            default: Default value if key doesn't exist

        Returns:
            The value for the key or default if not found
        """
        with self._lock:
            return self._state.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        """Get a copy of the entire state.

        Returns:
            A copy of the current state dictionary
        """
        with self._lock:
            return self._state.copy()

    def get_last_updated(self) -> datetime:
        """Get the timestamp of the last update.

        Returns:
            Datetime of the last state update
        """
        with self._lock:
            return self._last_updated


class DashboardService:
    """
    Main service for creating and managing secure, real-time, cross-cloud dashboards.

    This service orchestrates:
    - Data source fetching from configured sources
    - Differential privacy application
    - Neural preprocessing of data
    - Visualization generation using MLX backend
    - Insight generation using MLX language models
    - AR/VR integration
    - State management and updates
    - Background refresh and secure handling

    Attributes:
        config: Dashboard configuration
        state: Thread-safe dashboard state
        data_sources: Registered data sources
        visualization_backend: Visualization generation backend
        insight_generator: ML-based insight generator
        dp_engine: Differential privacy engine
        preprocessor: Neural data preprocessor
        credential_manager: Secure credential management
        encryptor: Data encryption/decryption handler
        ar_vr_manager: AR/VR integration manager
    """

    def __init__(
        self,
        config: Dict[str, Any],
        update_frequency: Union[UpdateFrequency, int] = UpdateFrequency.HOURLY,
        auto_start: bool = False,
    ):
        """
        Initialize the dashboard service.

        Args:
            config: Configuration dictionary for the dashboard
            update_frequency: How often to refresh data (enum or seconds)
            auto_start: Whether to start the update loop automatically

        Raises:
            DashboardConfigError: If configuration is invalid
        """
        self.logger = setup_logger("dashboard_service")
        self.logger.info("Initializing Dashboard Service")

        # Validate configuration
        self._validate_config(config)
        self.config = config

        # Initialize state
        self.state = DashboardState()

        # Set up security components
        self.credential_manager = CredentialManager()
        self.encryptor = Encryptor(self.credential_manager.get_encryption_key())

        # Set up data processing components
        self.data_sources = self._initialize_data_sources()
        self.dp_engine = DifferentialPrivacyEngine(
            epsilon=config.get("dp_config", {}).get("epsilon", 1.0),
            delta=config.get("dp_config", {}).get("delta", 1e-5),
        )
        self.preprocessor = NeuralPreprocessor()

        # Set up visualization components
        self.visualization_backend = MLXVisualizationBackend()

        # Set up insight generation
        self.insight_generator = InsightGenerator()

        # Set up AR/VR integration
        self.ar_vr_manager = ARVRIntegrationManager(config.get("ar_vr_config", {}))

        # Set up update mechanism
        self._setup_update_mechanism(update_frequency)
        self._stop_event = threading.Event()
        self._update_thread = None

        # Store initial metadata
        self.state.update("dashboard_name", config.get("name", "Unnamed Dashboard"))
        self.state.update("created_at", datetime.now())

        if auto_start:
            self.start()

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the dashboard configuration.

        Args:
            config: Configuration to validate

        Raises:
            DashboardConfigError: If configuration is invalid
        """
        required_keys = ["name", "data_sources"]

        for key in required_keys:
            if key not in config:
                raise DashboardConfigError(f"Missing required configuration key: {key}")

        if not isinstance(config["data_sources"], list) or not config["data_sources"]:
            raise DashboardConfigError("data_sources must be a non-empty list")

    def _initialize_data_sources(self) -> List[BaseDataSource]:
        """
        Initialize data sources from configuration.

        Returns:
            List of initialized data sources

        Raises:
            DataSourceError: If a data source cannot be initialized
        """
        registry = DataSourceRegistry()
        sources = []

        for source_config in self.config["data_sources"]:
            try:
                source_type = source_config["type"]
                source_class = registry.get_source_class(source_type)

                # Handle encrypted credentials if present
                if "credentials" in source_config and source_config.get(
                    "credentials_encrypted", False
                ):
                    source_config["credentials"] = self.encryptor.decrypt(
                        source_config["credentials"]
                    )

                sources.append(source_class(**source_config))
            except Exception as e:
                raise DataSourceError(f"Failed to initialize data source: {str(e)}")

        return sources

    def _setup_update_mechanism(self, update_frequency: Union[UpdateFrequency, int]) -> None:
        """
        Set up the update frequency for the dashboard.

        Args:
            update_frequency: How often to refresh data (enum or seconds)
        """
        if isinstance(update_frequency, UpdateFrequency):
            if update_frequency == UpdateFrequency.REALTIME:
                self.update_interval = 1  # 1 second for "realtime"
            elif update_frequency == UpdateFrequency.HOURLY:
                self.update_interval = 3600
            elif update_frequency == UpdateFrequency.DAILY:
                self.update_interval = 86400
            elif update_frequency == UpdateFrequency.WEEKLY:
                self.update_interval = 604800
            elif update_frequency == UpdateFrequency.MONTHLY:
                self.update_interval = 2592000
            else:  # CUSTOM or any other
                self.update_interval = self.config.get("custom_update_interval", 3600)
        else:
            # Assume it's a custom interval in seconds
            self.update_interval = update_frequency

        self.logger.info(f"Update interval set to {self.update_interval} seconds")

    def start(self) -> None:
        """
        Start the dashboard update loop in a background thread.
        """
        if self._update_thread and self._update_thread.is_alive():
            self.logger.warning("Update thread already running")
            return

        self._stop_event.clear()
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
        self.logger.info("Dashboard update thread started")

    def stop(self) -> None:
        """
        Stop the dashboard update loop.
        """
        self._stop_event.set()
        if self._update_thread:
            self._update_thread.join(timeout=10)
            if self._update_thread.is_alive():
                self.logger.warning("Update thread did not terminate properly")
            else:
                self.logger.info("Dashboard update thread stopped")

    def _update_loop(self) -> None:
        """
        Main update loop that runs in the background thread.
        """
        while not self._stop_event.is_set():
            try:
                self.refresh()
            except Exception as e:
                self.logger.error(f"Error in update loop: {str(e)}")

            # Wait for the next update interval or until stop is called
            self._stop_event.wait(self.update_interval)

    def refresh(self) -> None:
        """
        Perform a full refresh of all dashboard data and visualizations.

        This method orchestrates the complete data flow:
        1. Fetch data from all sources
        2. Apply differential privacy
        3. Preprocess data
        4. Generate visualizations
        5. Generate insights
        6. Update dashboard state
        7. Push to AR/VR clients if configured
        """
        self.logger.info("Starting dashboard refresh")

        # Step 1: Fetch data from all sources
        raw_data = self._fetch_data()

        # Step 2: Apply differential privacy
        dp_data = self._apply_differential_privacy(raw_data)

        # Step 3: Preprocess data
        processed_data = self._preprocess_data(dp_data)

        # Step 4: Generate visualizations
        visualizations = self._generate_visualizations(processed_data)

        # Step 5: Generate insights
        insights = self._generate_insights(processed_data, visualizations)

        # Step 6: Update dashboard state
        self._update_state(
            raw_data=raw_data,
            processed_data=processed_data,
            visualizations=visualizations,
            insights=insights,
        )

        # Step 7: Push to AR/VR clients if configured
        if self.config.get("enable_ar_vr", False):
            self._push_to_ar_vr()

        self.logger.info("Dashboard refresh completed")

    def _fetch_data(self) -> Dict[str, Any]:
        """
        Fetch data from all configured data sources.

        Returns:
            Dictionary of data keyed by source ID

        Raises:
            DataSourceError: If data fetch fails
        """
        self.logger.info("Fetching data from sources")
        result = {}

        for source in self.data_sources:
            try:
                source_id = source.get_id()
                data = source.fetch_data()

                # Apply encryption if source requires it
                if source.requires_encryption():
                    data = self.encryptor.encrypt(data)

                result[source_id] = data
            except Exception as e:
                self.logger.error(f"Error fetching from source {source.get_id()}: {str(e)}")
                # Continue with other sources rather than failing completely

        return result

    def _apply_differential_privacy(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply differential privacy to sensitive data.

        Args:
            raw_data: Raw data from sources

        Returns:
            Data with differential privacy applied
        """
        self.logger.info("Applying differential privacy")
        result = {}

        for source_id, data in raw_data.items():
            source = next((s for s in self.data_sources if s.get_id() == source_id), None)

            # Skip if source not found (shouldn't happen)
            if not source:
                continue

            # Skip if source doesn't require DP
            if not source.requires_differential_privacy():
                result[source_id] = data
                continue

            # Decrypt if necessary
            if source.requires_encryption():
                try:
                    data = self.encryptor.decrypt(data)
                except Exception as e:
                    self.logger.error(f"Error decrypting data from {source_id}: {str(e)}")
                    continue

            # Apply DP
            try:
                dp_data = self.dp_engine.apply_privacy(data, source.get_privacy_config())
                result[source_id] = dp_data
            except Exception as e:
                self.logger.error(f"Error applying DP to {source_id}: {str(e)}")
                # Skip this source if DP fails

        return result

    def _preprocess_data(self, dp_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess data using neural preprocessing.

        Args:
            dp_data: Data with differential privacy applied

        Returns:
            Preprocessed data ready for visualization
        """
        self.logger.info("Preprocessing data")
        result = {}

        for source_id, data in dp_data.items():
            try:
                preprocessed = self.preprocessor.preprocess(
                    data,
                    source_id=source_id,
                    config=self.config.get("preprocessing_config", {}),
                )
                result[source_id] = preprocessed
            except Exception as e:
                self.logger.error(f"Error preprocessing {source_id}: {str(e)}")
                # Use original data if preprocessing fails
                result[source_id] = data

        return result

    def _generate_visualizations(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate visualizations from processed data.

        Args:
            processed_data: Preprocessed data

        Returns:
            Dictionary of visualization elements
        """
        self.logger.info("Generating visualizations")
        visualization_config = self.config.get("visualization_config", {})

        try:
            return self.visualization_backend.generate_visualizations(
                processed_data, visualization_config
            )
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
            return {}

    def _generate_insights(
        self, processed_data: Dict[str, Any], visualizations: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate insights from processed data and visualizations.

        Args:
            processed_data: Preprocessed data
            visualizations: Generated visualizations

        Returns:
            List of insight objects
        """
        self.logger.info("Generating insights")
        insight_config = self.config.get("insight_config", {})

        try:
            return self.insight_generator.generate_insights(
                processed_data, visualizations, insight_config
            )
        except Exception as e:
            self.logger.error(f"Error generating insights: {str(e)}")
            return []

    def _update_state(
        self,
        raw_data: Dict[str, Any],
        processed_data: Dict[str, Any],
        visualizations: Dict[str, Any],
        insights: List[Dict[str, Any]],
    ) -> None:
        """
        Update the dashboard state with new data.

        Args:
            raw_data: Raw data from sources
            processed_data: Preprocessed data
            visualizations: Generated visualizations
            insights: Generated insights
        """
        self.state.update("last_raw_data", raw_data)
        self.state.update("processed_data", processed_data)
        self.state.update("visualizations", visualizations)
        self.state.update("insights", insights)
        self.state.update("last_refreshed", datetime.now())

    def _push_to_ar_vr(self) -> None:
        """
        Push current state to AR/VR clients.
        """
        try:
            current_state = self.state.get_all()
            self.ar_vr_manager.push_update(current_state)
        except Exception as e:
            self.logger.error(f"Error pushing to AR/VR: {str(e)}")

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current dashboard state.

        Returns:
            Current dashboard state
        """
        return self.state.get_all()

    def get_visualization(self, viz_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific visualization by ID.

        Args:
            viz_id: Visualization identifier

        Returns:
            Visualization data or None if not found
        """
        visualizations = self.state.get("visualizations", {})
        return visualizations.get(viz_id)

    def create_visualization_element(
        self, element_type: str, data_source_id: str, config: Dict[str, Any]
    ) -> str:
        """
        Create a new visualization element.

        Args:
            element_type: Type of visualization element
            data_source_id: Source data to visualize
            config: Visualization configuration

        Returns:
            ID of the created visualization element

        Raises:
            VisualizationError: If creation fails
        """
        # Get the current processed data for the source
        processed_data = self.state.get("processed_data", {}).get(data_source_id)

        if not processed_data:
            raise VisualizationError(f"No processed data for source {data_source_id}")

        # Create the visualization element
        try:
            element_id = self.visualization_backend.create_element(
                element_type, processed_data, config
            )

            # Update visualizations in state
            visualizations = self.state.get("visualizations", {})
            new_element = self.visualization_backend.get_element(element_id)
            visualizations[element_id] = new_element
            self.state.update("visualizations", visualizations)

            return element_id
        except Exception as e:
            raise VisualizationError(f"Failed to create visualization: {str(e)}")

    def update_visualization_element(self, element_id: str, config: Dict[str, Any]) -> None:
        """
        Update an existing visualization element.

        Args:
            element_id: ID of the element to update
            config: New configuration

        Raises:
            VisualizationError: If update fails
        """
        try:
            self.visualization_backend.update_element(element_id, config)

            # Update visualizations in state
            visualizations = self.state.get("visualizations", {})
            updated_element = self.visualization_backend.get_element(element_id)
            visualizations[element_id] = updated_element
            self.state.update("visualizations", visualizations)
        except Exception as e:
            raise VisualizationError(f"Failed to update visualization: {str(e)}")

    def delete_visualization_element(self, element_id: str) -> None:
        """
        Delete a visualization element.

        Args:
            element_id: ID of the element to delete

        Raises:
            VisualizationError: If deletion fails
        """
        try:
            self.visualization_backend.delete_element(element_id)

            # Update visualizations in state
            visualizations = self.state.get("visualizations", {})
            if element_id in visualizations:
                del visualizations[element_id]
                self.state.update("visualizations", visualizations)
        except Exception as e:
            raise VisualizationError(f"Failed to delete visualization: {str(e)}")

    def get_insights(self) -> List[Dict[str, Any]]:
        """
        Get all generated insights.

        Returns:
            List of insight objects
        """
        return self.state.get("insights", [])

    def export_dashboard(self, format_type: str = "json") -> Dict[str, Any]:
        """
        Export the dashboard for external use.

        Args:
            format_type: Export format (json, csv, etc.)

        Returns:
            Exported dashboard data
        """
        state = self.state.get_all()

        # Remove sensitive/internal data
        if "last_raw_data" in state:
            del state["last_raw_data"]

        # Format conversion would happen here based on format_type

        return state
