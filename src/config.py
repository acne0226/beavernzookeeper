"""
Configuration module for the personal work assistant daemon.
Loads settings from .env and defines constants.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")

# ── SSL certificate fix for macOS + Homebrew Python ──────────────────────────
# Python installed via Homebrew doesn't find system CA certs by default.
# Point SSL to certifi's bundle so all HTTPS calls (Slack, Google, etc.) work.
if not os.environ.get("SSL_CERT_FILE"):
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        pass

# ── Slack ──────────────────────────────────────────────────────────────────────
SLACK_BOT_TOKEN: str = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN: str = os.environ["SLACK_APP_TOKEN"]
SLACK_TARGET_USER_ID: str = os.environ["SLACK_TARGET_USER_ID"]

# ── Google OAuth2 ──────────────────────────────────────────────────────────────
GOOGLE_CLIENT_ID: str = os.environ["GOOGLE_CLIENT_ID"]
GOOGLE_CLIENT_SECRET: str = os.environ["GOOGLE_CLIENT_SECRET"]
GOOGLE_REFRESH_TOKEN: str = os.environ["GOOGLE_REFRESH_TOKEN"]
GOOGLE_TOKEN_FILE: Path = ROOT_DIR / "google_token.json"
GOOGLE_CREDENTIALS_FILE: Path = ROOT_DIR / "google_credentials.json"
GMAIL_TOKEN_FILE: Path = ROOT_DIR / "gmail_token.json"
CALENDAR_CACHE_FILE: Path = ROOT_DIR / "calendar_history_cache.json"

# ── Notion ─────────────────────────────────────────────────────────────────────
NOTION_TOKEN: str = os.environ["NOTION_TOKEN"]
NOTION_DB_ID: str = os.environ["NOTION_DB_ID"]

# ── Anthropic / Claude ─────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.environ["ANTHROPIC_API_KEY"]

# ── Web Search (optional) ──────────────────────────────────────────────────────
# Used by src/ai/web_search.py for EXTERNAL_FIRST meeting briefings.
# If not set, the web-search section falls back to Claude's built-in
# web_search tool (using ANTHROPIC_API_KEY above).
TAVILY_API_KEY: str = os.environ.get("TAVILY_API_KEY", "")

# ── Investment Team ────────────────────────────────────────────────────────────
# 11 internal kakaoventures team members; used to distinguish external vs internal
INVESTMENT_TEAM_EMAILS: list[str] = [
    "hyewon.anne@kakaoventures.co.kr",
    "invest1@kakaoventures.co.kr",
    "invest2@kakaoventures.co.kr",
    "invest3@kakaoventures.co.kr",
    "invest4@kakaoventures.co.kr",
    "invest5@kakaoventures.co.kr",
    "invest6@kakaoventures.co.kr",
    "invest7@kakaoventures.co.kr",
    "invest8@kakaoventures.co.kr",
    "invest9@kakaoventures.co.kr",
    "invest10@kakaoventures.co.kr",
]
INTERNAL_DOMAIN: str = "kakaoventures.co.kr"

# ── Calendar settings ──────────────────────────────────────────────────────────
# How many minutes ahead to look for upcoming meetings
MEETING_LOOKAHEAD_MINUTES: int = 15
# How far back to scan history for external meeting classification
CALENDAR_HISTORY_LOOKBACK_DAYS: int = 365
# Google Calendar primary calendar id
PRIMARY_CALENDAR_ID: str = "primary"

# ── Scheduler settings ─────────────────────────────────────────────────────────
# Polling interval in seconds for the meeting check job
SCHEDULER_POLL_INTERVAL_SECONDS: int = 60

# ── Retry settings ─────────────────────────────────────────────────────────────
API_RETRY_ATTEMPTS: int = 3
API_RETRY_DELAY_SECONDS: int = 10

# ── Slack channel search keywords ─────────────────────────────────────────────
SLACK_PRIORITY_CHANNEL_KEYWORDS: list[str] = ["투자", "squad-service"]
