"""Подключение к LLM через OpenRouter (OpenAI-совместимый API).

Ключ берётся из переменной окружения OPENROUTER_API_KEY
(на Streamlit Cloud — из Secrets). Модель можно поменять через OPENROUTER_MODEL.
"""
import json
import os
import re

from openai import OpenAI

BASE_URL = "https://openrouter.ai/api/v1"

# "openrouter/free" сам выбирает доступную бесплатную модель;
# остальные — запасные варианты, если роутер не ответил.
DEFAULT_MODELS = [
    "openrouter/free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
]


class LLMUnavailable(RuntimeError):
    pass


def is_configured() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def _models() -> list[str]:
    custom = os.environ.get("OPENROUTER_MODEL", "").strip()
    return [custom] + DEFAULT_MODELS if custom else DEFAULT_MODELS


def chat(system: str, user: str, temperature: float = 0.3, max_tokens: int = 2500) -> tuple[str, str]:
    """Возвращает (ответ, имя модели). Перебирает модели, пока одна не ответит."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise LLMUnavailable("Не задан OPENROUTER_API_KEY")

    client = OpenAI(base_url=BASE_URL, api_key=key, timeout=90, max_retries=1)
    errors = []
    for model in _models():
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers={"X-Title": "Psy Research Agents"},
            )
            text = (resp.choices[0].message.content or "").strip()
            # некоторые "думающие" модели оставляют рассуждения в <think>...</think>
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
            if text:
                return text, resp.model or model
            errors.append(f"{model}: пустой ответ")
        except Exception as exc:  # noqa: BLE001 — пробуем следующую модель
            errors.append(f"{model}: {type(exc).__name__}: {exc}"[:300])
    raise LLMUnavailable("Ни одна модель не ответила:\n" + "\n".join(errors))


def extract_json(text: str) -> dict:
    """Достаёт первый JSON-объект из ответа модели (даже если он в ```json ... ```)."""
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("В ответе модели нет JSON")
    return json.loads(match.group(0))
