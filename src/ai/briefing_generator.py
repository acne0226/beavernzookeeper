"""
AI-powered Per-Meeting Briefing Generator (Sub-AC 3b).

Uses the Anthropic Claude API to produce a structured, narrative briefing for
a single upcoming meeting from the raw context aggregated by
``MeetingContextAggregator``.

The generator enriches the rule-based ``meeting_briefing_formatter.py`` output
with AI-synthesised content in three dedicated sections:

1. **Attendee Bios** — concise professional context for each external attendee
   based on their email domain, display name, and past-meeting history.
2. **Agenda Prep** — suggested talking points, questions, and preparation notes
   derived from the meeting title, description, and email/Notion context.
3. **Email Thread Summary** — condensed overview of the most relevant email
   exchanges, highlighting key decisions, open items, and tone.

Design principles
-----------------
* **Accuracy-first**: Claude is prompted to only report information present in
  the provided context.  When context is insufficient, Claude is instructed to
  write "확인 불가" rather than speculating.
* **Retry policy**: Each Claude API call is retried up to
  ``API_RETRY_ATTEMPTS`` (3) times with ``API_RETRY_DELAY_SECONDS`` (10 s)
  between attempts, mirroring the global project retry policy.
* **Graceful degradation**: If the AI call fails after all retries,
  ``AIBriefingSections.ai_available`` is set to False and the formatter falls
  back to the rule-based representation — the briefing is still sent.
* **Token budget**: Prompts are capped so the total response stays within the
  per-call token budget (~1 000 tokens).  Long context items (email bodies,
  Notion records) are truncated before being passed to Claude.

Usage::

    from src.ai.briefing_generator import BriefingGenerator

    generator = BriefingGenerator()
    sections = generator.generate(raw_content)
    # sections.attendee_bios   – dict[email, bio_text]
    # sections.agenda_prep     – str (AI prep notes)
    # sections.email_summary   – str (AI email thread summary)
    # sections.ai_available    – bool
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.briefing.context_aggregator import RawBriefingContent

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Claude model to use for briefing generation
_CLAUDE_MODEL = "claude-haiku-4-5"

# Maximum tokens to request in Claude's response
_MAX_RESPONSE_TOKENS: int = 1024

# Max characters of email body included in prompt per thread
_MAX_EMAIL_BODY_IN_PROMPT: int = 400

# Max characters of Notion record content included in prompt per record
_MAX_NOTION_CONTENT_IN_PROMPT: int = 300

# Max email threads passed to Claude
_MAX_THREADS_IN_PROMPT: int = 5

# Max Notion records passed to Claude
_MAX_NOTION_IN_PROMPT: int = 5

# Retry policy (mirrors global API_RETRY_ATTEMPTS / API_RETRY_DELAY_SECONDS)
_RETRY_ATTEMPTS: int = 3
_RETRY_DELAY: float = 10.0


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class AIBriefingSections:
    """
    AI-generated sections to supplement the rule-based briefing formatter.

    Each field holds a rendered Slack *mrkdwn* string (or an empty string
    when the section could not be generated).  The formatter renders these
    as additional Slack Block Kit section blocks inserted between the
    attendee list and the email-thread sections.

    Attributes
    ----------
    attendee_bios:
        Maps external attendee email → short professional bio paragraph
        (≤ 3 sentences, Korean or English).  Empty string when not generated.
    agenda_prep:
        AI-generated preparation notes: suggested talking points and
        questions.  Markdown-formatted for Slack (bullet list).
    email_summary:
        Condensed summary of the most relevant email threads, highlighting
        key decisions and open items.
    ai_available:
        False when all Claude API retries were exhausted; the formatter
        should fall back to the rule-based representation in this case.
    error:
        Human-readable error description when ai_available=False.
    """

    attendee_bios: dict[str, str] = field(default_factory=dict)
    agenda_prep: str = ""
    email_summary: str = ""
    ai_available: bool = True
    error: Optional[str] = None

    @property
    def has_content(self) -> bool:
        """True if at least one AI section contains non-empty content."""
        return bool(self.attendee_bios or self.agenda_prep or self.email_summary)

    def to_dict(self) -> dict:
        return {
            "attendee_bios": self.attendee_bios,
            "agenda_prep": self.agenda_prep,
            "email_summary": self.email_summary,
            "ai_available": self.ai_available,
            "error": self.error,
        }


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_attendee_bio_prompt(
    email: str,
    display_name: str,
    company_domain: str,
    past_meeting_count: int,
    past_meeting_titles: list[str],
    web_search_snippet: str = "",
) -> str:
    """
    Build a single-attendee bio generation prompt for Claude.

    Returns a Korean/English mixed prompt suitable for a single-turn
    message to Claude.
    """
    name_line = display_name or email
    domain_line = f"소속 도메인: {company_domain}" if company_domain else ""
    history_line = (
        f"과거 미팅 횟수: {past_meeting_count}회"
        if past_meeting_count > 0
        else "과거 미팅 이력 없음 (첫 만남)"
    )
    past_titles_line = ""
    if past_meeting_titles:
        titles_text = ", ".join(f'"{t}"' for t in past_meeting_titles[:3])
        past_titles_line = f"과거 미팅 주제: {titles_text}"
    web_line = f"\n웹 검색 스니펫:\n{web_search_snippet[:500]}" if web_search_snippet else ""

    context_parts = [
        f"이름/이메일: {name_line}",
        domain_line,
        history_line,
        past_titles_line,
        web_line,
    ]
    context_text = "\n".join(p for p in context_parts if p)

    return (
        "다음 정보를 바탕으로 이 외부 미팅 참석자에 대한 간략한 프로필을 작성해 주세요.\n"
        "제공된 정보만 활용하고, 모르는 정보는 '확인 불가'라고 표시하세요.\n"
        "2~3문장 이내로 작성하며, 불렛 포인트(•) 없이 단락 형식으로 작성하세요.\n\n"
        f"[참석자 정보]\n{context_text}\n\n"
        "프로필:"
    )


def _build_agenda_prep_prompt(
    meeting_title: str,
    meeting_description: str,
    external_attendees: list[str],
    gmail_threads_summary: str,
    notion_records_summary: str,
) -> str:
    """
    Build an agenda preparation prompt for Claude.

    Returns a prompt that asks Claude to generate bullet-point prep notes.
    """
    desc_part = (
        f"미팅 설명:\n{meeting_description[:600]}"
        if meeting_description
        else "미팅 설명: (없음)"
    )
    attendees_part = (
        f"외부 참석자: {', '.join(external_attendees)}" if external_attendees else ""
    )
    email_part = (
        f"관련 이메일 스레드 요약:\n{gmail_threads_summary[:800]}"
        if gmail_threads_summary
        else ""
    )
    notion_part = (
        f"관련 Notion 딜/포트폴리오:\n{notion_records_summary[:500]}"
        if notion_records_summary
        else ""
    )

    context_parts = [
        f"미팅 제목: {meeting_title}",
        desc_part,
        attendees_part,
        email_part,
        notion_part,
    ]
    context_text = "\n\n".join(p for p in context_parts if p)

    return (
        "다음 정보를 바탕으로 미팅 준비 사항을 작성해 주세요.\n"
        "제공된 정보만 활용하고, 불확실한 내용은 포함하지 마세요.\n"
        "불렛 포인트 형식(• 항목)으로 3~5개 핵심 준비 사항을 작성하세요.\n\n"
        f"[미팅 컨텍스트]\n{context_text}\n\n"
        "준비 사항:"
    )


def _build_email_summary_prompt(threads_context: str) -> str:
    """
    Build an email thread summary prompt for Claude.

    Returns a prompt that asks Claude to produce a concise email exchange summary.
    """
    return (
        "다음 이메일 스레드들을 검토하고 간결한 요약을 작성해 주세요.\n"
        "주요 논의 내용, 결정 사항, 미해결 사항(액션 아이템)을 중심으로 작성하세요.\n"
        "제공된 정보만 활용하고, 추측은 하지 마세요.\n"
        "5문장 이내로 작성하세요.\n\n"
        f"[이메일 스레드]\n{threads_context}\n\n"
        "요약:"
    )


# ── Context serialisers ────────────────────────────────────────────────────────

def _summarise_gmail_threads(raw_content: "RawBriefingContent") -> str:
    """
    Build a compact text representation of Gmail threads for the prompt.

    Only includes subject, latest date, and truncated body of each thread.
    """
    if not raw_content.gmail_threads:
        return ""

    lines: list[str] = []
    for i, thread in enumerate(raw_content.gmail_threads[:_MAX_THREADS_IN_PROMPT], 1):
        subject = getattr(thread, "subject", "(제목 없음)") or "(제목 없음)"
        date_str = ""
        latest_date = getattr(thread, "latest_date", None)
        if latest_date:
            date_str = f" ({latest_date.strftime('%Y-%m-%d')})"
        msg_count = getattr(thread, "message_count", 0)

        # Try to get body text from first message
        body_text = ""
        messages = getattr(thread, "messages", [])
        if messages:
            first_msg = messages[0]
            body_text = getattr(first_msg, "body_text", "") or getattr(first_msg, "snippet", "")
            if body_text:
                body_text = f"\n   내용 요약: {body_text[:_MAX_EMAIL_BODY_IN_PROMPT]}"

        lines.append(
            f"{i}. [{subject}]{date_str} ({msg_count}개 메시지){body_text}"
        )

    return "\n".join(lines)


def _summarise_notion_records(raw_content: "RawBriefingContent") -> str:
    """
    Build a compact text representation of Notion records for the prompt.
    """
    if not raw_content.notion_records:
        return ""

    lines: list[str] = []
    for rec in raw_content.notion_records[:_MAX_NOTION_IN_PROMPT]:
        title = getattr(rec, "title", "") or getattr(rec, "company_name", "") or "(이름 없음)"
        status = getattr(rec, "status", "")
        status_part = f" [{status}]" if status else ""
        lines.append(f"• {title}{status_part}")

    return "\n".join(lines)


def _summarise_web_search(raw_content: "RawBriefingContent") -> str:
    """
    Extract the top web search snippet for attendee bio enrichment.
    Returns empty string when no web search results are available.
    """
    if raw_content.web_search_summary is None:
        return ""
    ws = raw_content.web_search_summary
    if not ws.has_results:
        return ""
    # Return the highest-scored snippet
    top = sorted(ws.results, key=lambda r: r.score, reverse=True)[0]
    return top.snippet


# ── Claude API caller ─────────────────────────────────────────────────────────

def _call_claude(
    client,
    prompt: str,
    max_tokens: int = _MAX_RESPONSE_TOKENS,
) -> str:
    """
    Send a single-turn message to Claude and return the response text.

    Raises on API errors (caller handles retry).
    """
    response = client.messages.create(
        model=_CLAUDE_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text_parts = [
        block.text
        for block in response.content
        if hasattr(block, "text") and block.text
    ]
    return " ".join(text_parts).strip()


def _call_claude_with_retry(
    client,
    prompt: str,
    label: str = "claude",
    max_tokens: int = _MAX_RESPONSE_TOKENS,
) -> tuple[str, Optional[str]]:
    """
    Call Claude with up to _RETRY_ATTEMPTS retries.

    Returns (response_text, error_message).
    error_message is None on success; response_text is "" on permanent failure.
    """
    last_error: Optional[str] = None

    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            text = _call_claude(client, prompt, max_tokens=max_tokens)
            return text, None
        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                "[BriefingGenerator/%s] Attempt %d/%d failed: %s",
                label,
                attempt,
                _RETRY_ATTEMPTS,
                exc,
            )
            if attempt < _RETRY_ATTEMPTS:
                time.sleep(_RETRY_DELAY)

    logger.error(
        "[BriefingGenerator/%s] All %d attempts failed. Last error: %s",
        label,
        _RETRY_ATTEMPTS,
        last_error,
    )
    return "", last_error


# ── Main generator ────────────────────────────────────────────────────────────

class BriefingGenerator:
    """
    AI-powered briefing generator that enriches meeting briefings with
    Claude-synthesised content.

    Each call to :meth:`generate` makes up to three Claude API calls:
    one for attendee bios (batched), one for agenda prep, and one for
    the email thread summary.  API failures in any section are non-fatal
    and annotated so the formatter can degrade gracefully.

    Args:
        api_key: Anthropic API key.  Loaded from ``src.config`` when None.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        if api_key is None:
            api_key = _load_config_key("ANTHROPIC_API_KEY")
        self._api_key: Optional[str] = api_key or None
        self._client = None

    # ── Lazy client initialisation ─────────────────────────────────────────────

    def _ensure_client(self) -> Optional[object]:
        """
        Lazy-initialise the Anthropic client.

        Returns the client on success, None when the SDK is unavailable or
        the API key is not set.
        """
        if self._client is not None:
            return self._client

        if not self._api_key:
            logger.warning(
                "[BriefingGenerator] ANTHROPIC_API_KEY not set — AI sections disabled"
            )
            return None

        try:
            import anthropic  # type: ignore[import]
            self._client = anthropic.Anthropic(api_key=self._api_key)
            return self._client
        except ImportError as exc:
            logger.error(
                "[BriefingGenerator] anthropic package not installed: %s", exc
            )
            return None

    @property
    def is_available(self) -> bool:
        """True if the Anthropic client is initialised and the API key is set."""
        return self._ensure_client() is not None

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(
        self,
        raw_content: "RawBriefingContent",
        generate_bios: bool = True,
        generate_agenda: bool = True,
        generate_email_summary: bool = True,
    ) -> AIBriefingSections:
        """
        Generate AI-enriched briefing sections for a single meeting.

        Parameters
        ----------
        raw_content:
            The ``RawBriefingContent`` produced by ``MeetingContextAggregator``.
        generate_bios:
            When True (default), generate attendee bio text for each external
            attendee.  Set False to skip (reduces API calls).
        generate_agenda:
            When True (default), generate agenda preparation notes.
        generate_email_summary:
            When True (default), generate an email thread summary.
            Only rendered when gmail_threads is non-empty.

        Returns
        -------
        AIBriefingSections — always returns, never raises.
        ``ai_available=False`` when all Claude calls failed.
        """
        client = self._ensure_client()

        if client is None:
            return AIBriefingSections(
                ai_available=False,
                error="Anthropic client unavailable (check ANTHROPIC_API_KEY)",
            )

        logger.info(
            "[BriefingGenerator] Generating AI sections for '%s' "
            "(bios=%s agenda=%s email=%s)",
            raw_content.meeting_title,
            generate_bios,
            generate_agenda,
            generate_email_summary,
        )

        sections = AIBriefingSections()
        any_success = False
        errors: list[str] = []

        # ── 1. Attendee bios ───────────────────────────────────────────────────
        if generate_bios and raw_content.external_attendees:
            # Use a single batched call for all attendees to conserve tokens
            bio_results, bio_error = self._generate_attendee_bios(
                raw_content, client
            )
            if bio_error:
                errors.append(f"attendee_bios: {bio_error}")
            else:
                sections.attendee_bios = bio_results
                any_success = True

        # ── 2. Agenda preparation ──────────────────────────────────────────────
        if generate_agenda:
            agenda_text, agenda_error = self._generate_agenda_prep(
                raw_content, client
            )
            if agenda_error:
                errors.append(f"agenda_prep: {agenda_error}")
            else:
                sections.agenda_prep = agenda_text
                any_success = True

        # ── 3. Email thread summary ────────────────────────────────────────────
        if generate_email_summary and raw_content.gmail_threads and raw_content.gmail_available:
            email_text, email_error = self._generate_email_summary(
                raw_content, client
            )
            if email_error:
                errors.append(f"email_summary: {email_error}")
            else:
                sections.email_summary = email_text
                any_success = True
        elif generate_email_summary and not raw_content.gmail_threads:
            # No emails to summarise — not an error
            logger.debug(
                "[BriefingGenerator] Skipping email summary — no Gmail threads"
            )

        # ── Finalise ───────────────────────────────────────────────────────────
        if errors and not any_success:
            sections.ai_available = False
            sections.error = "; ".join(errors)
            logger.error(
                "[BriefingGenerator] All AI sections failed for '%s': %s",
                raw_content.meeting_title,
                sections.error,
            )
        elif errors:
            # Partial failure — still usable
            logger.warning(
                "[BriefingGenerator] Partial AI failure for '%s': %s",
                raw_content.meeting_title,
                "; ".join(errors),
            )

        logger.info(
            "[BriefingGenerator] Done for '%s': bios=%d agenda=%s email=%s "
            "ai_available=%s",
            raw_content.meeting_title,
            len(sections.attendee_bios),
            bool(sections.agenda_prep),
            bool(sections.email_summary),
            sections.ai_available,
        )
        return sections

    # ── Section generators ─────────────────────────────────────────────────────

    def _generate_attendee_bios(
        self,
        raw_content: "RawBriefingContent",
        client,
    ) -> tuple[dict[str, str], Optional[str]]:
        """
        Generate professional bios for all external attendees in one call.

        Returns (bio_dict, error_message) where bio_dict maps
        attendee_email → bio_text.  error_message is None on success.
        """
        external = raw_content.external_attendees
        if not external:
            return {}, None

        web_snippet = _summarise_web_search(raw_content)

        # Build a combined prompt for all attendees to save API calls
        attendee_blocks: list[str] = []
        for profile in external:
            bio_prompt_part = (
                f"### {profile.display_name or profile.email}\n"
                f"이메일: {profile.email}\n"
                f"소속 도메인: {profile.company_domain or '알 수 없음'}\n"
                f"과거 미팅: {profile.past_meeting_count}회"
            )
            if profile.past_meeting_titles:
                titles = ", ".join(f'"{t}"' for t in profile.past_meeting_titles[:2])
                bio_prompt_part += f"\n과거 미팅 주제: {titles}"
            attendee_blocks.append(bio_prompt_part)

        attendees_text = "\n\n".join(attendee_blocks)
        web_section = (
            f"\n\n[웹 검색 정보]\n{web_snippet[:500]}" if web_snippet else ""
        )

        prompt = (
            "다음 미팅 참석자들의 간략한 프로필을 작성해 주세요.\n"
            "제공된 정보만 활용하고, 불확실한 내용은 '확인 불가'라고 표시하세요.\n"
            "각 참석자에 대해 2~3문장으로 작성하세요.\n"
            "응답 형식: JSON 객체로 이메일을 키, 프로필 텍스트를 값으로 작성하세요.\n"
            "예시: {\"ceo@company.com\": \"홍길동 씨는 company.com 소속입니다. ...\"}\\n\n"
            f"[참석자 목록]\n{attendees_text}{web_section}\n\n"
            "JSON 응답:"
        )

        response_text, error = _call_claude_with_retry(
            client, prompt, label="attendee_bios"
        )

        if error:
            return {}, error

        # Parse JSON response
        bio_dict = _parse_bio_json(response_text, external)
        return bio_dict, None

    def _generate_agenda_prep(
        self,
        raw_content: "RawBriefingContent",
        client,
    ) -> tuple[str, Optional[str]]:
        """
        Generate agenda preparation notes for the meeting.

        Returns (agenda_text, error_message).
        """
        external_emails = [
            p.display_name or p.email for p in raw_content.external_attendees
        ]
        gmail_summary = _summarise_gmail_threads(raw_content)
        notion_summary = _summarise_notion_records(raw_content)

        prompt = _build_agenda_prep_prompt(
            meeting_title=raw_content.meeting_title,
            meeting_description=raw_content.meeting_description or "",
            external_attendees=external_emails,
            gmail_threads_summary=gmail_summary,
            notion_records_summary=notion_summary,
        )

        response_text, error = _call_claude_with_retry(
            client, prompt, label="agenda_prep"
        )

        if error:
            return "", error

        return response_text.strip(), None

    def _generate_email_summary(
        self,
        raw_content: "RawBriefingContent",
        client,
    ) -> tuple[str, Optional[str]]:
        """
        Generate a concise summary of the relevant email threads.

        Returns (summary_text, error_message).
        """
        threads_context = _summarise_gmail_threads(raw_content)
        if not threads_context:
            return "", None

        prompt = _build_email_summary_prompt(threads_context)

        response_text, error = _call_claude_with_retry(
            client, prompt, label="email_summary"
        )

        if error:
            return "", error

        return response_text.strip(), None


# ── JSON parsing helper ────────────────────────────────────────────────────────

def _parse_bio_json(
    response_text: str,
    external_profiles,
) -> dict[str, str]:
    """
    Attempt to parse a JSON bio response from Claude.

    Falls back gracefully when Claude's output is not valid JSON by
    assigning the entire response as the bio for the first attendee (or
    splitting by newline when multiple attendees are present).

    Args:
        response_text: Raw text response from Claude.
        external_profiles: List of AttendeeProfile for fallback key generation.

    Returns:
        dict mapping attendee_email → bio_text.
    """
    # Try strict JSON parse first
    text = response_text.strip()
    # Claude sometimes wraps JSON in ```json ... ```
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            # Validate keys look like emails (or at least strings)
            return {str(k): str(v) for k, v in parsed.items()}
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: assign full text as a combined bio for the first attendee
    if external_profiles:
        first_email = external_profiles[0].email
        return {first_email: response_text.strip()}

    return {}


# ── Config helper ─────────────────────────────────────────────────────────────

def _load_config_key(attr: str) -> Optional[str]:
    """Safely load a string config attribute; return None on any failure."""
    try:
        import importlib
        config = importlib.import_module("src.config")
        value = getattr(config, attr, None)
        return value if value else None
    except Exception:  # pylint: disable=broad-except
        return None
