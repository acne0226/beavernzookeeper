# Slack integration package
from .bot import WorkAssistantBot
from .history_retriever import SlackHistoryRetriever, SlackMessage, SlackChannel, SlackHistoryResult

__all__ = [
    "WorkAssistantBot",
    "SlackHistoryRetriever",
    "SlackMessage",
    "SlackChannel",
    "SlackHistoryResult",
]
