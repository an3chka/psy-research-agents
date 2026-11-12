"""Агент-планировщик: превращает вопрос пользователя на русском в поисковые запросы для источников."""
from dataclasses import dataclass

from . import llm

# Готовые темы: работают даже без нейросети.
PRESETS = {
    "Депрессия": {
        "pubmed_query": '("depressive disorder"[MeSH Terms] OR depression[Title/Abstract]) AND (therapy OR treatment OR psychotherapy)',
        "keywords_en": "depression treatment",
        "trials_condition": "Depression",
        "trials_term": "",
    },
    "Тревожность": {
        "topic_ru": "Тревожные расстройства",
        "pubmed_query": '("anxiety disorders"[MeSH Terms] OR anxiety[Title/Abstract]) AND (therapy OR treatment OR psychotherapy)',
        "keywords_en": "anxiety disorder treatment",
        "trials_condition": "Anxiety Disorders",
        "trials_term": "",
    },
    "ПТСР": {
        "pubmed_query": '("stress disorders, post-traumatic"[MeSH Terms] OR PTSD[Title/Abstract]) AND (therapy OR treatment)',
        "keywords_en": "PTSD treatment",
        "trials_condition": "PTSD",
        "trials_term": "",
    },
    "Выгорание": {
        "topic_ru": "Эмоциональное выгорание",
        "pubmed_query": '("burnout, psychological"[MeSH Terms] OR burnout[Title/Abstract]) AND (intervention OR prevention OR treatment)',
        "keywords_en": "burnout intervention",
        "trials_condition": "Burnout",
        "trials_term": "",
    },
    "Бессонница": {
        "pubmed_query": '("sleep initiation and maintenance disorders"[MeSH Terms] OR insomnia[Title/Abstract]) AND (cognitive behavioral therapy OR treatment)',
        "keywords_en": "insomnia treatment",
        "trials_condition": "Insomnia",
        "trials_term": "",
    },
    "СДВГ": {
        "pubmed_query": '("attention deficit disorder with hyperactivity"[MeSH Terms] OR ADHD[Title/Abstract]) AND (therapy OR treatment)',
        "keywords_en": "ADHD treatment",
        "trials_condition": "ADHD",
        "trials_term": "",
    },
    "РПП": {
        "topic_ru": "Расстройства пищевого поведения",
        "pubmed_query": '("feeding and eating disorders"[MeSH Terms] OR "eating disorder"[Title/Abstract]) AND (therapy OR treatment)',
        "keywords_en": "eating disorders treatment",
        "trials_condition": "Eating Disorders",
        "trials_term": "",
    },
    "КПТ": {
        "topic_ru": "Когнитивно-поведенческая терапия",
        "pubmed_query": '"cognitive behavioral therapy"[MeSH Terms] OR "cognitive behavioural therapy"[Title/Abstract]',
        "keywords_en": "cognitive behavioral therapy",
        "trials_condition": "",
        "trials_term": "cognitive behavioral therapy",
    },
}

SYSTEM_PROMPT = """Ты — агент-планировщик в системе поиска научной информации по психологии и психиатрии.
Пользователь задаёт вопрос на русском. Составь поисковые запросы на английском для двух источников:
1) PubMed (синтаксис E-utilities: можно MeSH Terms, Title/Abstract, AND/OR, кавычки);
2) ClinicalTrials.gov API v2: condition — заболевание/состояние, term — дополнительные слова (метод терапии, препарат), может быть пустым.
Ответь ТОЛЬКО JSON-объектом без пояснений:
{"topic_ru": "короткое название темы на русском",
 "pubmed_query": "...",
 "keywords_en": "2-4 простых английских слова для запасного поиска",
 "trials_condition": "...",
 "trials_term": "...",
 "is_psychology": true/false}"""


@dataclass
class SearchPlan:
    topic_ru: str
    pubmed_query: str
    keywords_en: str
    trials_condition: str
    trials_term: str
    is_psychology: bool = True
    source: str = "preset"  # preset | llm | raw
    note: str = ""


def plan(user_query: str, use_llm: bool = True) -> SearchPlan:
    query = user_query.strip()

    if query in PRESETS:
        return SearchPlan(source="preset", **{"topic_ru": query, **PRESETS[query]})

    if use_llm and llm.is_configured():
        try:
            text, model = llm.chat(SYSTEM_PROMPT, query, temperature=0.0, max_tokens=600)
            return _from_llm(llm.extract_json(text), query, f"модель: {model}")
        except Exception as exc:  # noqa: BLE001
            note = f"нейросеть не сработала ({type(exc).__name__}), ищу по исходному тексту"
    elif not use_llm:
        note = "нейросеть выключена (не введён пароль), ищу по исходному тексту"
    else:
        note = "нейросеть не подключена, ищу по исходному тексту"

    # Запасной вариант: отправляем запрос как есть (лучше работает на английском).
    return SearchPlan(
        topic_ru=query,
        pubmed_query=query,
        keywords_en=query,
        trials_condition="",
        trials_term=query,
        source="raw",
        note=note,
    )


def _from_llm(data: dict, query: str, note: str) -> SearchPlan:
    return SearchPlan(
        topic_ru=str(data.get("topic_ru") or query),
        pubmed_query=str(data.get("pubmed_query") or data.get("keywords_en") or query),
        keywords_en=str(data.get("keywords_en") or query),
        trials_condition=str(data.get("trials_condition") or ""),
        trials_term=str(data.get("trials_term") or ""),
        is_psychology=bool(data.get("is_psychology", True)),
        source="llm",
        note=note,
    )


def replan(user_query: str, previous: SearchPlan, problem: str, advice: str) -> SearchPlan | None:
    """Переписывает запросы по замечанию агента-критика. None — если нейросеть не ответила."""
    user = (f"Вопрос пользователя: {user_query}\n\n"
            f"Прошлые запросы:\nPubMed: {previous.pubmed_query}\n"
            f"ClinicalTrials: condition={previous.trials_condition!r}, term={previous.trials_term!r}\n\n"
            f"Агент-критик проверил результаты и пишет: {problem}\nСовет: {advice}\n\n"
            "Составь исправленные запросы с учётом замечания. Формат ответа тот же.")
    try:
        text, model = llm.chat(SYSTEM_PROMPT, user, temperature=0.0, max_tokens=600)
        plan_ = _from_llm(llm.extract_json(text), previous.topic_ru, f"исправлено по замечанию критика, модель: {model}")
        plan_.topic_ru = previous.topic_ru
        return plan_
    except Exception:  # noqa: BLE001
        return None
