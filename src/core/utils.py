"""
Lab 11 — Helper Utilities
"""
import asyncio
import os

from core.config import get_llm_provider, PROVIDER_OPENROUTER  # noqa: F401
from core.openai_runtime import OpenAIRunner

# Lỗi tạm thời phía provider (quá tải / quota phút) → đợi rồi thử lại.
# Lỗi key (401/403) không nằm đây: thử lại vô ích.
_TRANSIENT_MARKERS = (
    "503", "unavailable", "high demand", "overloaded",
    "429", "resource_exhausted", "rate limit", "rate_limit",
    "502", "504", "timeout", "timed out", "connection",
)


def is_transient_error(exc: BaseException | None) -> bool:
    """True nếu lỗi (hoặc lỗi gốc trong chuỗi __cause__/__context__) là tạm thời."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        msg = f"{type(exc).__name__} {exc}".lower()
        if any(m in msg for m in _TRANSIENT_MARKERS):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


async def chat_with_agent(agent, runner, user_message: str, session_id=None):
    """Send a message to the agent and get the response.

    Works with OpenAIRunner (OpenAI Red / OpenRouter Blue) and Google ADK (Gemini Red).
    Retries with backoff on transient provider errors (503 high demand, 429).
    Số lần thử: env ``LAB_LLM_RETRIES`` (mặc định 4; đợi 10s, 20s, 40s, 60s).
    """
    retries = int(os.environ.get("LAB_LLM_RETRIES", "4") or 0)
    delay = 10.0
    for attempt in range(retries + 1):
        try:
            # Lần thử lại dùng session mới: session cũ có thể giữ lượt hỏi dở.
            return await _chat_once(
                agent, runner, user_message, session_id if attempt == 0 else None
            )
        except Exception as e:  # noqa: BLE001
            if attempt >= retries or not is_transient_error(e):
                raise
            print(
                f"  (provider bận — {type(e).__name__}; thử lại "
                f"{attempt + 1}/{retries} sau {delay:.0f}s)"
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)
    raise RuntimeError("unreachable")  # pragma: no cover


async def _chat_once(agent, runner, user_message: str, session_id=None):
    provider = getattr(runner, "provider", None)
    if isinstance(runner, OpenAIRunner) or provider in ("openrouter", "openai"):
        text = await runner.chat(agent, user_message)
        return text, None

    from google.genai import types

    user_id = "student"
    app_name = runner.app_name

    session = None
    if session_id is not None:
        try:
            session = await runner.session_service.get_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
        except (ValueError, KeyError):
            pass

    if session is None:
        try:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )
        except Exception:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    final_response = ""
    async for event in runner.run_async(
        user_id=user_id, session_id=session.id, new_message=content
    ):
        if hasattr(event, "content") and event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    final_response += part.text

    return final_response, session
