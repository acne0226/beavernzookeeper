"""Notion integration package."""
from src.notion.notion_client import (
    NotionClient,
    NotionRecord,
    DatabaseSchema,
    NotionPageContent,
    NotionPageSection,
)
from src.notion.portfolio_cache import (
    PortfolioCache,
    PortfolioCompany,
    EmailMatchResult,
    get_portfolio_cache,
)

__all__ = [
    # notion_client
    "NotionClient",
    "NotionRecord",
    "DatabaseSchema",
    "NotionPageContent",
    "NotionPageSection",
    # portfolio_cache
    "PortfolioCache",
    "PortfolioCompany",
    "EmailMatchResult",
    "get_portfolio_cache",
]
