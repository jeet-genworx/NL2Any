"""HTTP client for fetching table and column descriptions from the schema documentation API."""

import logging
import os
import httpx
from backend.src.schemas.metadata import TableDescriptionResponse

logger = logging.getLogger(__name__)


class DescriptionsClient:
    """Client for retrieving table and column descriptions from the REST API."""

    def __init__(
        self,
        base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
        ).rstrip("/")
        self._client = client
        self.timeout = timeout

    async def get_table_description(
        self,
        database_type: str,
        table_name: str,
    ) -> TableDescriptionResponse | None:
        """Fetch table and column descriptions for one table.

        Calls GET /descriptions/{database_type}/{table_name}.
        Returns None if not found or if the call fails.
        """
        url = f"{self.base_url}/descriptions/{database_type}/{table_name}"
        try:
            if self._client:
                response = await self._client.get(url, timeout=self.timeout)
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(url)

            if response.status_code == 200:
                return TableDescriptionResponse.model_validate(response.json())
            elif response.status_code == 404:
                logger.warning(
                    "Table description not found for %s.%s (404)",
                    database_type,
                    table_name,
                )
                return None
            else:
                logger.warning(
                    "Descriptions API returned status %d for %s.%s: %s",
                    response.status_code,
                    database_type,
                    table_name,
                    response.text,
                )
                return None
        except Exception as err:
            logger.warning(
                "Error connecting to Descriptions API at %s: %s",
                url,
                err,
            )
            return None
