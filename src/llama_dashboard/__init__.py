"""
Llama Dashboard: Web UI for the LlamaAI Ecosystem.

Provides visualization and interaction points for monitoring and
using Llama services.
"""

__version__ = "0.1.0"

from .core import DashboardService
from .data_sources import (  # AWSDataSource # Uncomment when implemented
    AzureDataSource,
    BaseDataSource,
    DataSourceRegistry,
    GCPDataSource,
)

__all__ = [
    "__version__",
    "DashboardService",
    "BaseDataSource",
    "DataSourceRegistry",
    "GCPDataSource",
    "AzureDataSource",
    # "AWSDataSource",
]
