# 🧠 PsyResearch: мультиагентный поиск по психологии и психиатрии

Учебный проект. Веб-приложение на Streamlit, в котором несколько агентов собирают информацию из двух источников:

- **PubMed** (NCBI E-utilities API): опубликованные научные статьи;
- **ClinicalTrials.gov** (API v2): клинические испытания.

## Агенты

| Агент | Что делает |
|---|---|
| 🧭 Планировщик | Превращает вопрос на русском в поисковые запросы для каждого источника (LLM или готовые темы) |
| 📚 Агент PubMed | Ищет статьи, загружает аннотации, считает число публикаций по годам |
| 🧪 Агент ClinicalTrials | Ищет испытания: статус, фаза, тип вмешательства, страны, число участников |
| 📊 Статистик | Тренд публикаций (темп роста, линейный тренд), распределения, доли с 95% ДИ Уилсона, медиана и IQR |
| 🧠 Аналитик | Пишет сводку на русском со ссылками [PMID] и [NCT]; приложение проверяет, что каждая ссылка есть в найденных данных |

Агенты PubMed и ClinicalTrials работают параллельно. Если нейросеть недоступна, сводка собирается по шаблону.

## Запуск локально

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # вставить ключ OpenRouter
.venv/bin/streamlit run app.py
```

## Деплой на Streamlit Community Cloud

1. Загрузить репозиторий на GitHub.
2. На [share.streamlit.io](https://share.streamlit.io) нажать **Create app**, выбрать репозиторий и указать файл `app.py`.
3. В **Advanced settings → Secrets** вставить `OPENROUTER_API_KEY = "sk-or-v1-..."`.

## Структура

```
app.py                  — веб-интерфейс
agents/orchestrator.py  — порядок запуска агентов и журнал их работы
agents/planner.py       — планировщик и готовые темы
agents/pubmed_agent.py  — агент PubMed
agents/trials_agent.py  — агент ClinicalTrials.gov
agents/stats_agent.py   — статистик
agents/analyst.py       — аналитик (LLM)
agents/llm.py           — подключение к OpenRouter
```

*Не является медицинской консультацией.*
