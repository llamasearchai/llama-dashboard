#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data sources module for the Llama Dashboard.

This module provides interfaces and implementations for various data sources
that can be used with the Llama Dashboard.
"""

import json
import os
from abc import ABC, abstractmethod
from io import BytesIO, StringIO
from typing import Any, Dict, List, Optional, Type, Union

import pandas as pd


class DataSourceError(Exception):
    """Exception raised when a data source operation fails."""

    pass


class BaseDataSource(ABC):
    """Abstract base class for all data sources.

    Data sources are responsible for fetching data from external systems,
    handling authentication, and returning the data in a standardized format.

    Attributes:
        id: Unique identifier for the data source
        name: Human-readable name
        requires_differential_privacy: Whether differential privacy should be applied
        requires_encryption: Whether data should be encrypted at rest
        privacy_config: Configuration for differential privacy
    """

    def __init__(
        self,
        name: str,
        requires_differential_privacy: bool = False,
        requires_encryption: bool = False,
        privacy_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize a data source.

        Args:
            name: Human-readable name for the data source
            requires_differential_privacy: Whether DP should be applied
            requires_encryption: Whether data should be encrypted
            privacy_config: Configuration for differential privacy
            **kwargs: Additional configuration options
        """
        self.name = name
        self.requires_differential_privacy = requires_differential_privacy
        self.requires_encryption = requires_encryption
        self.privacy_config = privacy_config or {}
        self.id = kwargs.get("id", name.lower().replace(" ", "_"))

    @abstractmethod
    def fetch_data(self) -> Union[pd.DataFrame, Dict[str, Any], List[Any]]:
        """Fetch data from the source.

        Returns:
            The fetched data in a standardized format

        Raises:
            DataSourceError: If the fetch operation fails
        """
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get the schema of the data.

        Returns:
            A dictionary describing the schema of the data

        Raises:
            DataSourceError: If the schema cannot be determined
        """
        pass


class DataSourceRegistry:
    """Registry of available data source types.

    This registry maps data source type names to their implementation classes,
    allowing for dynamic instantiation of data sources by type name.
    """

    _instance = None
    _registry: Dict[str, Type[BaseDataSource]] = {}

    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(DataSourceRegistry, cls).__new__(cls)
        return cls._instance

    def register_source(self, source_type: str, source_class: Type[BaseDataSource]) -> None:
        """Register a data source type.

        Args:
            source_type: Identifier for the source type
            source_class: Class implementing BaseDataSource
        """
        self._registry[source_type] = source_class

    def get_source_class(self, source_type: str) -> Type[BaseDataSource]:
        """Get the class for a data source type.

        Args:
            source_type: Identifier for the source type

        Returns:
            The class implementing this data source type

        Raises:
            ValueError: If the source type is not registered
        """
        if source_type not in self._registry:
            raise ValueError(f"Data source type '{source_type}' is not registered")
        return self._registry[source_type]

    def get_available_sources(self) -> Dict[str, Type[BaseDataSource]]:
        """Get all registered data source types.

        Returns:
            Dictionary mapping source type names to their classes
        """
        return self._registry.copy()


class GCPDataSource(BaseDataSource):
    """Data source for Google Cloud Platform services.

    Supports various GCP data services including BigQuery, Cloud Storage, and Firestore.

    Attributes:
        service: GCP service to use
        project_id: GCP project ID
        credentials: GCP credentials
        service_config: Service-specific configuration
    """

    def __init__(
        self,
        name: str,
        service: str,
        project_id: str,
        credentials: Optional[Dict[str, Any]] = None,
        service_config: Optional[Dict[str, Any]] = None,
        requires_differential_privacy: bool = False,
        requires_encryption: bool = True,
        privacy_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize a GCP data source.

        Args:
            name: Human-readable name
            service: GCP service to use (bigquery, storage, firestore, etc.)
            project_id: GCP project ID
            credentials: GCP credentials
            service_config: Service-specific configuration
            requires_differential_privacy: Whether DP should be applied
            requires_encryption: Whether data should be encrypted
            privacy_config: Configuration for differential privacy
            **kwargs: Additional configuration options
        """
        super().__init__(
            name=name,
            requires_differential_privacy=requires_differential_privacy,
            requires_encryption=requires_encryption,
            privacy_config=privacy_config,
            **kwargs,
        )

        self.service = service.lower()
        self.project_id = project_id
        self.credentials = credentials or {}
        self.service_config = service_config or {}

        # Set up GCP clients based on service
        try:
            if self.service == "bigquery":
                from google.cloud import bigquery
                from google.oauth2 import service_account

                if isinstance(self.credentials, dict):
                    # Create credentials from dict
                    gcp_credentials = service_account.Credentials.from_service_account_info(
                        self.credentials
                    )
                elif isinstance(self.credentials, str) and os.path.exists(self.credentials):
                    # Create credentials from file
                    gcp_credentials = service_account.Credentials.from_service_account_file(
                        self.credentials
                    )
                else:
                    # Use default credentials
                    gcp_credentials = None

                self.client = bigquery.Client(project=self.project_id, credentials=gcp_credentials)

            elif self.service == "storage":
                from google.cloud import storage
                from google.oauth2 import service_account

                if isinstance(self.credentials, dict):
                    gcp_credentials = service_account.Credentials.from_service_account_info(
                        self.credentials
                    )
                elif isinstance(self.credentials, str) and os.path.exists(self.credentials):
                    gcp_credentials = service_account.Credentials.from_service_account_file(
                        self.credentials
                    )
                else:
                    gcp_credentials = None

                self.client = storage.Client(project=self.project_id, credentials=gcp_credentials)

            elif self.service == "firestore":
                from google.cloud import firestore
                from google.oauth2 import service_account

                if isinstance(self.credentials, dict):
                    gcp_credentials = service_account.Credentials.from_service_account_info(
                        self.credentials
                    )
                elif isinstance(self.credentials, str) and os.path.exists(self.credentials):
                    gcp_credentials = service_account.Credentials.from_service_account_file(
                        self.credentials
                    )
                else:
                    gcp_credentials = None

                self.client = firestore.Client(project=self.project_id, credentials=gcp_credentials)
            else:
                raise DataSourceError(f"Unsupported GCP service: {self.service}")

        except ImportError as e:
            # Determine which package is missing
            if "google.cloud" in str(e):
                missing = self.service
            else:
                missing = "google-cloud-core"

            raise DataSourceError(
                f"google-cloud-{missing} is required for GCP {self.service} data sources"
            )
        except Exception as e:
            raise DataSourceError(f"Error initializing GCP {self.service} client: {str(e)}")

    def fetch_data(self) -> Union[pd.DataFrame, Dict[str, Any], List[Any]]:
        """Fetch data from the GCP service.

        Returns:
            Data from the GCP service

        Raises:
            DataSourceError: If data fetch fails
        """
        try:
            if self.service == "bigquery":
                return self._fetch_from_bigquery()
            elif self.service == "storage":
                return self._fetch_from_storage()
            elif self.service == "firestore":
                return self._fetch_from_firestore()
            else:
                raise DataSourceError(f"Unsupported GCP service: {self.service}")
        except Exception as e:
            raise DataSourceError(f"Error fetching from GCP {self.service}: {str(e)}")

    def _fetch_from_bigquery(self) -> pd.DataFrame:
        """Fetch data from BigQuery.

        Returns:
            DataFrame with query results

        Raises:
            DataSourceError: If BigQuery fetch fails
        """
        query = self.service_config.get("query")

        if not query:
            raise DataSourceError("BigQuery query is required")

        try:
            # Run the query
            query_job = self.client.query(query)

            # Convert to DataFrame
            return query_job.to_dataframe()
        except Exception as e:
            raise DataSourceError(f"BigQuery fetch failed: {str(e)}")

    def _fetch_from_storage(self) -> Union[pd.DataFrame, Dict[str, Any], str]:
        """Fetch data from Cloud Storage.

        Returns:
            Blob contents

        Raises:
            DataSourceError: If Cloud Storage fetch fails
        """
        bucket_name = self.service_config.get("bucket")
        blob_name = self.service_config.get("blob")

        if not bucket_name or not blob_name:
            raise DataSourceError("Cloud Storage bucket and blob are required")

        try:
            # Get the bucket and blob
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            # Download as bytes
            content = blob.download_as_bytes()

            # Determine format from file extension
            file_ext = os.path.splitext(blob_name)[1].lower()

            if file_ext in [".csv", ".txt"]:
                return pd.read_csv(StringIO(content.decode("utf-8")))
            elif file_ext in [".json"]:
                return json.loads(content.decode("utf-8"))
            elif file_ext in [".parquet"]:
                return pd.read_parquet(BytesIO(content))
            elif file_ext in [".xlsx", ".xls"]:
                return pd.read_excel(BytesIO(content))
            else:
                # Return raw content
                return content
        except Exception as e:
            raise DataSourceError(f"Cloud Storage fetch failed: {str(e)}")

    def _fetch_from_firestore(self) -> List[Dict[str, Any]]:
        """Fetch data from Firestore.

        Returns:
            List of documents

        Raises:
            DataSourceError: If Firestore fetch fails
        """
        collection = self.service_config.get("collection")

        if not collection:
            raise DataSourceError("Firestore collection is required")

        try:
            # Get all documents in the collection
            docs = self.client.collection(collection).stream()

            # Convert to list of dicts
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            raise DataSourceError(f"Firestore fetch failed: {str(e)}")

    def get_schema(self) -> Dict[str, Any]:
        """Get the schema of the data.

        Returns:
            Schema definition

        Raises:
            DataSourceError: If schema inference fails
        """
        try:
            if self.service == "bigquery":
                query = self.service_config.get("query")

                if not query:
                    raise DataSourceError("BigQuery query is required")

                # Create a query job for the schema
                job_config = self.client._connection.query_class.QueryJobConfig()
                job_config.dry_run = True

                query_job = self.client.query(query, job_config=job_config)

                # Extract schema from job
                schema = {}
                for field in query_job.schema:
                    schema[field.name] = {
                        "type": field.field_type,
                        "mode": field.mode,
                        "description": field.description,
                    }

                return {"fields": schema}

            elif self.service == "storage":
                # For Cloud Storage, we need to fetch a sample of the data first
                sample_data = self._fetch_from_storage()

                if isinstance(sample_data, pd.DataFrame):
                    return {
                        "columns": list(sample_data.columns),
                        "dtypes": {col: str(dtype) for col, dtype in sample_data.dtypes.items()},
                    }
                elif isinstance(sample_data, dict):
                    return {
                        "type": "object",
                        "properties": {
                            k: {"type": type(v).__name__} for k, v in sample_data.items()
                        },
                    }
                elif isinstance(sample_data, list) and sample_data:
                    if isinstance(sample_data[0], dict):
                        return {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    k: {"type": type(v).__name__} for k, v in sample_data[0].items()
                                },
                            },
                        }
                    else:
                        return {
                            "type": "array",
                            "items": {"type": type(sample_data[0]).__name__},
                        }
                else:
                    return {"type": type(sample_data).__name__}

            elif self.service == "firestore":
                collection = self.service_config.get("collection")

                if not collection:
                    raise DataSourceError("Firestore collection is required")

                # Get a sample document
                docs = list(self.client.collection(collection).limit(1).stream())

                if not docs:
                    return {"type": "object", "properties": {}}

                sample_doc = docs[0].to_dict()

                return {
                    "type": "object",
                    "properties": {k: {"type": type(v).__name__} for k, v in sample_doc.items()},
                }

            else:
                return {"error": f"Schema inference not implemented for {self.service}"}
        except Exception as e:
            raise DataSourceError(f"Schema inference failed: {str(e)}")


class AzureDataSource(BaseDataSource):
    """Data source for Microsoft Azure services.

    Supports various Azure data services including Blob Storage, SQL Database, and Cosmos DB.

    Attributes:
        service: Azure service to use
        credentials: Azure credentials
        service_config: Service-specific configuration
    """

    def __init__(
        self,
        name: str,
        service: str,
        credentials: Optional[Dict[str, str]] = None,
        service_config: Optional[Dict[str, Any]] = None,
        requires_differential_privacy: bool = False,
        requires_encryption: bool = True,
        privacy_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize an Azure data source.

        Args:
            name: Human-readable name
            service: Azure service to use (blob, sql, cosmos, etc.)
            credentials: Azure credentials
            service_config: Service-specific configuration
            requires_differential_privacy: Whether DP should be applied
            requires_encryption: Whether data should be encrypted
            privacy_config: Configuration for differential privacy
            **kwargs: Additional configuration options
        """
        super().__init__(
            name=name,
            requires_differential_privacy=requires_differential_privacy,
            requires_encryption=requires_encryption,
            privacy_config=privacy_config,
            **kwargs,
        )

        self.service = service.lower()
        self.credentials = credentials or {}
        self.service_config = service_config or {}

        # Set up Azure clients based on service
        try:
            if self.service == "blob":
                from azure.storage.blob import BlobServiceClient

                connection_string = self.credentials.get("connection_string")
                account_key = self.credentials.get("account_key")
                account_name = self.credentials.get("account_name")

                if connection_string:
                    self.client = BlobServiceClient.from_connection_string(connection_string)
                elif account_name and account_key:
                    from azure.storage.blob import BlobServiceClient

                    conn_str = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
                    self.client = BlobServiceClient.from_connection_string(conn_str)
                else:
                    raise DataSourceError("Azure Blob Storage credentials missing")

            elif self.service == "sql":
                import pyodbc

                server = self.service_config.get("server")
                database = self.service_config.get("database")
                username = self.credentials.get("username")
                password = self.credentials.get("password")

                if not server or not database or not username or not password:
                    raise DataSourceError("Azure SQL connection parameters missing")

                self.connection_string = (
                    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                    f"SERVER={server};"
                    f"DATABASE={database};"
                    f"UID={username};"
                    f"PWD={password}"
                )

                # Test connection
                self.conn = pyodbc.connect(self.connection_string)
                self.conn.close()

            elif self.service == "cosmos":
                from azure.cosmos import CosmosClient

                endpoint = self.service_config.get("endpoint")
                key = self.credentials.get("key")

                if not endpoint or not key:
                    raise DataSourceError("Azure Cosmos DB connection parameters missing")

                self.client = CosmosClient(endpoint, key)

            else:
                raise DataSourceError(f"Unsupported Azure service: {self.service}")

        except ImportError as e:
            # Determine which package is missing
            if "azure" in str(e):
                if self.service == "blob":
                    missing = "azure-storage-blob"
                elif self.service == "cosmos":
                    missing = "azure-cosmos"
                else:
                    missing = "azure-core"
            elif "pyodbc" in str(e):
                missing = "pyodbc"
            else:
                missing = str(e)

            raise DataSourceError(f"{missing} is required for Azure {self.service} data sources")
        except Exception as e:
            raise DataSourceError(f"Error initializing Azure {self.service} client: {str(e)}")

    def fetch_data(self) -> Union[pd.DataFrame, Dict[str, Any], List[Any]]:
        """Fetch data from the Azure service.

        Returns:
            Data from the Azure service

        Raises:
            DataSourceError: If data fetch fails
        """
        try:
            if self.service == "blob":
                return self._fetch_from_blob()
            elif self.service == "sql":
                return self._fetch_from_sql()
            elif self.service == "cosmos":
                return self._fetch_from_cosmos()
            else:
                raise DataSourceError(f"Unsupported Azure service: {self.service}")
        except Exception as e:
            raise DataSourceError(f"Error fetching from Azure {self.service}: {str(e)}")

    def _fetch_from_blob(self) -> Union[pd.DataFrame, Dict[str, Any], str]:
        """Fetch data from Azure Blob Storage.

        Returns:
            Blob contents

        Raises:
            DataSourceError: If Blob Storage fetch fails
        """
        container_name = self.service_config.get("container")
        blob_name = self.service_config.get("blob")

        if not container_name or not blob_name:
            raise DataSourceError("Azure Blob Storage container and blob are required")

        try:
            # Get the container and blob clients
            container_client = self.client.get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob_name)

            # Download as bytes
            download_stream = blob_client.download_blob()
            content = download_stream.readall()

            # Determine format from file extension
            file_ext = os.path.splitext(blob_name)[1].lower()

            if file_ext in [".csv", ".txt"]:
                return pd.read_csv(StringIO(content.decode("utf-8")))
            elif file_ext in [".json"]:
                return json.loads(content.decode("utf-8"))
            elif file_ext in [".parquet"]:
                return pd.read_parquet(BytesIO(content))
            elif file_ext in [".xlsx", ".xls"]:
                return pd.read_excel(BytesIO(content))
            else:
                # Return raw content
                return content
        except Exception as e:
            raise DataSourceError(f"Azure Blob Storage fetch failed: {str(e)}")

    def _fetch_from_sql(self) -> pd.DataFrame:
        """Fetch data from Azure SQL Database.

        Returns:
            DataFrame with query results

        Raises:
            DataSourceError: If SQL Database fetch fails
        """
        query = self.service_config.get("query")

        if not query:
            raise DataSourceError("Azure SQL query is required")

        try:
            import pyodbc

            # Create a new connection
            conn = pyodbc.connect(self.connection_string)

            try:
                # Execute query and fetch results
                return pd.read_sql(query, conn)
            finally:
                conn.close()
        except Exception as e:
            raise DataSourceError(f"Azure SQL fetch failed: {str(e)}")

    def _fetch_from_cosmos(self) -> List[Dict[str, Any]]:
        """Fetch data from Azure Cosmos DB.

        Returns:
            List of documents

        Raises:
            DataSourceError: If Cosmos DB fetch fails
        """
        database_name = self.service_config.get("database")
        container_name = self.service_config.get("container")
        query = self.service_config.get("query", "SELECT * FROM c")

        if not database_name or not container_name:
            raise DataSourceError("Azure Cosmos DB database and container are required")

        try:
            # Get the database and container clients
            database_client = self.client.get_database_client(database_name)
            container_client = database_client.get_container_client(container_name)

            # Execute the query
            items = list(
                container_client.query_items(query=query, enable_cross_partition_query=True)
            )

            return items
        except Exception as e:
            raise DataSourceError(f"Azure Cosmos DB fetch failed: {str(e)}")

    def get_schema(self) -> Dict[str, Any]:
        """Get the schema of the data.

        Returns:
            Schema definition

        Raises:
            DataSourceError: If schema inference fails
        """
        try:
            if self.service == "blob":
                # For Blob Storage, we need to fetch a sample of the data first
                sample_data = self._fetch_from_blob()

                if isinstance(sample_data, pd.DataFrame):
                    return {
                        "columns": list(sample_data.columns),
                        "dtypes": {col: str(dtype) for col, dtype in sample_data.dtypes.items()},
                    }
                elif isinstance(sample_data, dict):
                    return {
                        "type": "object",
                        "properties": {
                            k: {"type": type(v).__name__} for k, v in sample_data.items()
                        },
                    }
                elif isinstance(sample_data, list) and sample_data:
                    if isinstance(sample_data[0], dict):
                        return {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    k: {"type": type(v).__name__} for k, v in sample_data[0].items()
                                },
                            },
                        }
                    else:
                        return {
                            "type": "array",
                            "items": {"type": type(sample_data[0]).__name__},
                        }
                else:
                    return {"type": type(sample_data).__name__}

            elif self.service == "sql":
                import pyodbc

                query = self.service_config.get("query")

                if not query:
                    raise DataSourceError("Azure SQL query is required")

                # Create a new connection
                conn = pyodbc.connect(self.connection_string)

                try:
                    # Modify query to return no rows but keep structure
                    schema_query = f"SELECT TOP 0 * FROM ({query}) AS subquery"

                    # Execute query
                    cursor = conn.cursor()
                    cursor.execute(schema_query)

                    # Extract column information
                    columns = [column[0] for column in cursor.description]
                    types = [column[1].__name__ for column in cursor.description]

                    return {"columns": columns, "types": dict(zip(columns, types))}
                finally:
                    conn.close()

            elif self.service == "cosmos":
                database_name = self.service_config.get("database")
                container_name = self.service_config.get("container")

                if not database_name or not container_name:
                    raise DataSourceError("Azure Cosmos DB database and container are required")

                # Get container properties
                database_client = self.client.get_database_client(database_name)
                container_client = database_client.get_container_client(container_name)
                container_props = container_client.read()

                # Get a sample document if available
                items = list(
                    container_client.query_items(
                        query="SELECT TOP 1 * FROM c", enable_cross_partition_query=True
                    )
                )

                if not items:
                    return {
                        "partition_key": container_props.get("partitionKey", {}).get("paths", []),
                        "properties": {},
                    }

                sample_doc = items[0]

                return {
                    "partition_key": container_props.get("partitionKey", {}).get("paths", []),
                    "properties": {k: {"type": type(v).__name__} for k, v in sample_doc.items()},
                }

            else:
                return {"error": f"Schema inference not implemented for {self.service}"}
        except Exception as e:
            raise DataSourceError(f"Schema inference failed: {str(e)}")


# Register the data source types
registry = DataSourceRegistry()
registry.register_source("gcp", GCPDataSource)
registry.register_source("azure", AzureDataSource)

# Add placeholder for AWS
# registry.register_source("aws", AWSDataSource)
