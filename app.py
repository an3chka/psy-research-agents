"""PsyResearch: мультиагентный сбор информации из PubMed и ClinicalTrials.gov по психологии и психиатрии."""
import os
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="PsyResearch — мультиагентный поиск", page_icon="🧠", layout="wide")

# Ключи из Streamlit Secrets → в переменные окружения (агенты читают их оттуда)
for _key in ("OPENROUTER_API_KEY", "OPENROUTER_MODEL", "NCBI_API_KEY"):
    try:
        if _key in st.secrets and not os.environ.get(_key):
            os.environ[_key] = str(st.secrets[_key])
    except Exception:  # noqa: BLE001 — secrets.toml может отсутствовать
        pass

from agents import llm, orchestrator  # noqa: E402
from agents.planner import PRESETS  # noqa: E402
from agents.stats_agent import PHASE_RU, STATUS_RU  # noqa: E402

ACCENT = "#2a78d6"
MUTED = "#9aa0a6"
THIS_YEAR = date.today().year


# ---------------------------------------------------------------- графики
def _layout(fig: go.Figure, height: int, x_title: str = "", y_title: str = "") -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=16, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        hoverlabel=dict(font_size=13),
        bargap=0.25,
    )
    fig.update_xaxes(title=x_title, showgrid=False, zeroline=False)
    fig.update_yaxes(title=y_title, gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


def ranked_bars(df: pd.DataFrame, label_col: str, value_col: str = "Количество", unit: str = "испытаний"):
    """Горизонтальные столбцы по убыванию, один цвет, подписи значений справа."""
    if df.empty:
        st.caption("Нет данных")
        return
    d = df.sort_values(value_col, ascending=True)
    fig = go.Figure(go.Bar(
        x=d[value_col], y=d[label_col], orientation="h",
        marker=dict(color=ACCENT, cornerradius=4),
        text=d[value_col], textposition="outside", cliponaxis=False,
        hovertemplate=f"%{{y}}<br>%{{x}} {unit}<extra></extra>",
    ))
    _layout(fig, height=max(180, 34 * len(d) + 40))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def years_line(years: pd.DataFrame):
    full = years[~years["Неполный год"]]
    partial = years[years["Неполный год"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=full["Год"], y=full["Публикаций"], mode="lines+markers",
        line=dict(color=ACCENT, width=2), marker=dict(size=8, color=ACCENT),
        hovertemplate="%{x}: %{y:,} публикаций<extra></extra>",
    ))
    if not partial.empty and not full.empty:
        bridge = pd.concat([full.tail(1), partial])
        fig.add_trace(go.Scatter(
            x=bridge["Год"], y=bridge["Публикаций"], mode="lines",
            line=dict(color=ACCENT, width=2, dash="dot"), hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=partial["Год"], y=partial["Публикаций"], mode="markers",
            marker=dict(size=8, color="white", line=dict(color=ACCENT, width=2)),
            hovertemplate="%{x}: %{y:,} публикаций (год ещё не закончился)<extra></extra>",
        ))
    _layout(fig, height=320, y_title="Публикаций за год")
    fig.update_xaxes(dtick=1)
    fig.update_layout(separators=", ")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def start_years_bars(df: pd.DataFrame):
    if df.empty:
        st.caption("Нет данных")
        return
    fig = go.Figure(go.Bar(
        x=df["Год старта"], y=df["Испытаний"], marker=dict(color=ACCENT, cornerradius=4),
        hovertemplate="%{x}: %{y} испытаний<extra></extra>",
    ))
    _layout(fig, height=260, y_title="Испытаний")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def fmt(n: int) -> str:
    return f"{n:,}".replace(",", " ")


# ---------------------------------------------------------------- боковая панель
with st.sidebar:
    st.header("⚙️ Настройки поиска")
    year_from, year_to = st.slider("Период", 2000, THIS_YEAR, (2015, THIS_YEAR))
    max_articles = st.slider("Сколько статей показать", 5, 30, 12)
    trials_sample = st.slider("Сколько испытаний анализировать", 50, 500, 200, step=50)
    only_active = st.checkbox("Только активные испытания (идёт набор / продолжаются)")
    with_trend = st.checkbox("Считать динамику публикаций по годам", value=True,
                             help="Отдельный запрос в PubMed на каждый год — добавляет несколько секунд")
    st.divider()
    if llm.is_configured():
        st.success("🧠 Нейросеть подключена (OpenRouter)")
    else:
        st.warning("🧠 Нейросеть не подключена — сводка будет собрана по шаблону. "
                   "Свободный ввод лучше писать на английском.")
    st.caption("Источники: [PubMed](https://pubmed.ncbi.nlm.nih.gov/) · "
               "[ClinicalTrials.gov](https://clinicaltrials.gov/)")

# ---------------------------------------------------------------- шапка и ввод
st.title("🧠 PsyResearch")
st.markdown(
    "Мультиагентный поиск научной информации по **психологии и психиатрии**. "
    "Агенты одновременно ищут опубликованные исследования в **PubMed** и клинические испытания "
    "в **ClinicalTrials.gov**, считают статистику и собирают общую сводку."
)

st.write("**Быстрый выбор темы:**")
preset_names = list(PRESETS)
cols = st.columns(4)
for i, name in enumerate(preset_names):
    if cols[i % 4].button(name, use_container_width=True):
        st.session_state["query"] = name
        st.session_state["run"] = True

with st.form("search"):
    query = st.text_input(
        "Или свой запрос",
        key="query",
        placeholder="например: эффективность mindfulness при тревоге у студентов",
    )
    if st.form_submit_button("🔎 Найти", type="primary"):
        st.session_state["run"] = True

# ---------------------------------------------------------------- запуск агентов
if st.session_state.pop("run", False) and st.session_state.get("query", "").strip():
    settings = orchestrator.Settings(
        max_articles=max_articles, trials_sample=trials_sample,
        year_from=year_from, year_to=year_to,
        only_active_trials=only_active, with_trend=with_trend,
    )
    with st.status("Агенты работают…", expanded=True) as status:
        result = orchestrator.run(
            st.session_state["query"].strip(), settings,
            on_step=lambda s: st.write(f"`{s.seconds:>4}с` **{s.agent}**: {s.message}"),
        )
        status.update(label="Готово ✅", state="complete", expanded=False)
    st.session_state["result"] = result

result: orchestrator.RunResult | None = st.session_state.get("result")
if result is None:
    st.info("Выберите тему или введите свой запрос — агенты найдут статьи и испытания.")
    st.stop()

# ---------------------------------------------------------------- результаты
for source, err in result.errors.items():
    st.error(f"**{source}**: {err}")
if not result.plan.is_psychology:
    st.info("Похоже, запрос не про психологию — приложение всё равно поищет, но оно настроено на эту тему.")

f = result.stats.facts
st.subheader(f"Тема: {result.plan.topic_ru}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Статей в PubMed", fmt(f.get("pubmed_total", 0)),
          help=f"За {year_from}–{year_to}")
m2.metric("Испытаний в ClinicalTrials", fmt(f.get("trials_total", 0)))
m3.metric("Активных испытаний", f"{f.get('trials_active_share', 0)}%", help="Доля в проанализированной выборке")
if "behavioral_share" in f:
    b = f["behavioral_share"]
    m4.metric("С психотерапией / тренингами", f"{b['pct']}%",
              help=f"Доля испытаний с поведенческим вмешательством, 95% ДИ {b['ci_low']}–{b['ci_high']}%, n={b['n']}")

tab_sum, tab_art, tab_tr, tab_stat, tab_agents = st.tabs(
    ["🧠 Сводка", "📚 Статьи", "🧪 Испытания", "📊 Статистика", "🤖 Как работают агенты"]
)

with tab_sum:
    s = result.summary
    st.markdown(s.text)
    st.divider()
    cap = f"Сводку написал агент-аналитик ({s.model}). Ссылок на источники: {len(s.cited)}."
    if s.unknown_ids:
        st.warning("⚠️ Аналитик сослался на источники, которых нет в найденных данных (возможная ошибка нейросети): "
                   + ", ".join(s.unknown_ids))
    elif s.cited:
        cap += " Все ссылки проверены — каждая есть среди найденных статей и испытаний ✅"
    st.caption(cap)
    if s.error:
        with st.expander("Почему не сработала нейросеть"):
            st.code(s.error)

with tab_art:
    articles = result.pubmed.articles if result.pubmed else []
    if not articles:
        st.info("Статьи не найдены.")
    else:
        st.caption(f"Поисковый запрос PubMed: `{result.pubmed.query_used}` · сортировка по релевантности")
        for a in articles:
            badges = " · ".join(t for t in a.pub_types if t != "Journal Article")
            with st.expander(f"**{a.title}** ({a.year or '—'})"):
                st.markdown(f"{a.authors} — *{a.journal}*  \n{badges}")
                st.write(a.abstract or "_Аннотации нет_")
                st.markdown(f"[Открыть в PubMed (PMID {a.pmid})]({a.url})")
        df_art = pd.DataFrame([{
            "PMID": a.pmid, "Год": a.year, "Название": a.title, "Журнал": a.journal,
            "Авторы": a.authors, "Тип": "; ".join(a.pub_types), "Ссылка": a.url,
        } for a in articles])
        st.download_button("⬇️ Скачать статьи (CSV)", df_art.to_csv(index=False).encode("utf-8-sig"),
                           "pubmed_articles.csv", "text/csv")

with tab_tr:
    trials = result.trials.trials if result.trials else []
    if not trials:
        st.info("Испытания не найдены.")
    else:
        df_tr = pd.DataFrame([{
            "NCT": t.url, "Название": t.title, "Статус": STATUS_RU.get(t.status, t.status),
            "Фаза": ", ".join(PHASE_RU.get(p, p) for p in t.phases),
            "Вмешательства": "; ".join(t.interventions[:3]),
            "Страны": ", ".join(t.countries[:3]), "Участников": t.enrollment, "Старт": t.start_year,
        } for t in trials])
        statuses = st.multiselect("Фильтр по статусу", sorted(df_tr["Статус"].unique()), placeholder="Все статусы")
        view = df_tr[df_tr["Статус"].isin(statuses)] if statuses else df_tr
        st.dataframe(
            view, use_container_width=True, hide_index=True, height=480,
            column_config={
                "NCT": st.column_config.LinkColumn("NCT", display_text=r"study/(NCT\d+)"),
                "Участников": st.column_config.NumberColumn(format="%d"),
                "Старт": st.column_config.NumberColumn(format="%d"),
            },
        )
        st.download_button("⬇️ Скачать испытания (CSV)", view.to_csv(index=False).encode("utf-8-sig"),
                           "clinical_trials.csv", "text/csv")

with tab_stat:
    st_res = result.stats
    if not st_res.years.empty:
        st.markdown("#### Сколько публикаций выходит по теме каждый год (PubMed)")
        years_line(st_res.years)
        notes = []
        if f.get("pubmed_cagr_pct") is not None:
            first, last = f["pubmed_first_last"]
            notes.append(f"За {f['pubmed_period']}: с {fmt(first)} до {fmt(last)} публикаций в год, "
                         f"средний темп роста **{f['pubmed_cagr_pct']}% в год**, "
                         f"линейный тренд **{f['pubmed_trend_slope']:+} публикаций в год**.")
        if st_res.years["Неполный год"].any():
            notes.append(f"{THIS_YEAR} год ещё не закончился (пунктир) — в расчёт тренда не входит.")
        st.caption(" ".join(notes))

    if result.trials and result.trials.trials:
        n = f.get("trials_sample", 0)
        st.markdown(f"#### Клинические испытания (выборка из {n} самых релевантных)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Тип вмешательства**")
            ranked_bars(st_res.interventions, "Тип вмешательства")
            st.caption("Одно испытание может включать несколько типов вмешательства.")
        with c2:
            st.markdown("**Статус**")
            ranked_bars(st_res.statuses, "Статус")
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Фаза**")
            ranked_bars(st_res.phases, "Фаза")
            st.caption("Фазы бывают только у испытаний лекарств; психотерапия обычно «без фазы».")
        with c4:
            st.markdown("**Страны (топ-12)**")
            ranked_bars(st_res.countries, "Страна")
        st.markdown("**Год начала испытаний**")
        start_years_bars(st_res.start_years)

        if "behavioral_share" in f:
            b, d = f["behavioral_share"], f["drug_share"]
            e = f.get("enrollment", {})
            st.markdown("#### Психотерапия или лекарства?")
            st.markdown(
                f"- Испытаний с **поведенческим** вмешательством (психотерапия, тренинги): **{b['pct']}%** "
                f"(95% доверительный интервал Уилсона: {b['ci_low']}–{b['ci_high']}%)\n"
                f"- Испытаний с **лекарствами**: **{d['pct']}%** (95% ДИ {d['ci_low']}–{d['ci_high']}%)\n"
                f"- n = {b['n']} интервенционных испытаний"
                + (f"\n- Размер выборки участников: медиана **{e['median']}**, "
                   f"межквартильный размах {e['q1']}–{e['q3']}" if e else "")
            )

with tab_agents:
    st.markdown("#### Схема системы")
    st.graphviz_chart("""
    digraph {
      rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fillcolor="#e8f0fb", color="#2a78d6", fontname="Helvetica"];
      edge [color="#7a7a7a"];
      user [label="👤 Запрос\\nпользователя", fillcolor="#f2f2f2", color="#9aa0a6"];
      planner [label="🧭 Планировщик\\n(LLM: перевод и\\nпоисковые запросы)"];
      pubmed [label="📚 Агент PubMed\\n(E-utilities API)"];
      trials [label="🧪 Агент ClinicalTrials\\n(API v2)"];
      stats [label="📊 Статистик\\n(тренды, доли, ДИ)"];
      analyst [label="🧠 Аналитик\\n(LLM: сводка\\nсо ссылками)"];
      ui [label="🖥 Streamlit", fillcolor="#f2f2f2", color="#9aa0a6"];
      user -> planner; planner -> pubmed; planner -> trials;
      pubmed -> stats; trials -> stats; pubmed -> analyst; trials -> analyst; stats -> analyst;
      analyst -> ui;
    }
    """)
    st.markdown(
        "- **Планировщик** переводит вопрос на русском в запросы на языке каждого источника.\n"
        "- **Агенты PubMed и ClinicalTrials** работают **параллельно**, каждый со своим API.\n"
        "- **Статистик** считает динамику публикаций, распределения и доверительные интервалы.\n"
        "- **Аналитик** пишет сводку только по найденным данным; приложение проверяет, "
        "что каждая ссылка в сводке реально есть среди результатов."
    )
    st.markdown("#### Журнал последнего запуска")
    st.dataframe(
        pd.DataFrame([{"Время, с": s.seconds, "Агент": s.agent, "Действие": s.message} for s in result.trace]),
        use_container_width=True, hide_index=True,
    )

st.divider()
st.caption("Учебный проект. Не является медицинской консультацией и не заменяет обращение к специалисту.")
