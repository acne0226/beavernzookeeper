"""Gmail integration package."""
from src.gmail.gmail_client import GmailClient, EmailMessage, EmailThread
from src.gmail.thread_fetcher import (
    ThreadSummary,
    get_threads_for_company,
    build_thread_summary,
    get_latest_thread_summary,
    get_all_thread_summaries,
)

__all__ = [
    # Core client and data models
    "GmailClient",
    "EmailMessage",
    "EmailThread",
    # Thread fetcher (Sub-AC 7.2)
    "ThreadSummary",
    "get_threads_for_company",
    "build_thread_summary",
    "get_latest_thread_summary",
    "get_all_thread_summaries",
]
