"""PsyResearch: мультиагентный сбор информации из PubMed и ClinicalTrials.gov по психологии и психиатрии."""
import hmac
import os
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="PsyResearch", page_icon=":material/psychology:", layout="wide")

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

ACCENT = "#2f8f62"  # нежно-зелёный, проверен на контраст и цветовую слепоту
INK = "#1c2a22"
MUTED = "#66756b"
LINE = "#d6e4da"
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
    fig.update_yaxes(title=y_title, gridcolor="rgba(28,42,34,0.08)", zeroline=False)
    fig.update_layout(font=dict(family="Inter, sans-serif", color=INK, size=13))
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
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


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
            marker=dict(size=9, color="white", line=dict(color=ACCENT, width=2)),
            hovertemplate="%{x}: %{y:,} публикаций (год ещё не закончился)<extra></extra>",
        ))
    _layout(fig, height=320, y_title="Публикаций за год")
    fig.update_xaxes(dtick=1)
    fig.update_layout(separators=", ")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def start_years_bars(df: pd.DataFrame):
    if df.empty:
        st.caption("Нет данных")
        return
    fig = go.Figure(go.Bar(
        x=df["Год старта"], y=df["Испытаний"], marker=dict(color=ACCENT, cornerradius=4),
        hovertemplate="%{x}: %{y} испытаний<extra></extra>",
    ))
    _layout(fig, height=260, y_title="Испытаний")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "") or os.environ.get(name, ""))
    except Exception:  # noqa: BLE001 — secrets.toml может отсутствовать
        return os.environ.get(name, "")


def fmt(n: int) -> str:
    return f"{n:,}".replace(",", " ")


# ---------------------------------------------------------------- стили
st.markdown("""
<style>
[data-testid="stDecoration"] {display: none;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {max-width: 1180px; padding-top: 2.4rem; padding-bottom: 4rem;}
h1, h2, h3, h4 {letter-spacing: -0.015em;}
.hero-kicker {font-size: .78rem; font-weight: 600; letter-spacing: .09em; text-transform: uppercase;
              color: #2f8f62; margin-bottom: .35rem;}
.hero-title {font-size: 2.7rem; font-weight: 700; line-height: 1.08; margin: 0 0 .55rem 0; color: #1c2a22;}
.hero-sub {font-size: 1.05rem; line-height: 1.55; color: #5d6c62; max-width: 760px; margin-bottom: 1.6rem;}
[data-testid="stForm"] {background: #ffffff; border: 1px solid #d6e4da; border-radius: 18px; padding: 1.1rem 1.25rem .4rem;}
[data-testid="stMetric"] {background: #ffffff; border: 1px solid #d6e4da; border-radius: 16px; padding: 16px 18px;}
[data-testid="stMetricLabel"] p {color: #5d6c62;}
[data-testid="stExpander"] details {background: #ffffff; border-color: #d6e4da; border-radius: 14px;}
.stTabs [data-baseweb="tab-list"] {gap: .4rem;}
.steps {display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-top: 1.4rem;}
.step {background: #ffffff; border: 1px solid #d6e4da; border-radius: 16px; padding: 18px 20px;}
.step b {display: block; font-size: 1rem; margin-bottom: .3rem; color: #1c2a22;}
.step span {color: #5d6c62; font-size: .93rem; line-height: 1.5;}
.step i {font-style: normal; color: #2f8f62; font-weight: 700; font-size: .85rem; display: block; margin-bottom: .4rem;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- боковая панель
with st.sidebar:
    st.subheader("Настройки поиска")
    year_from, year_to = st.slider("Период", 2000, THIS_YEAR, (2015, THIS_YEAR))
    max_articles = st.slider("Сколько статей показать", 5, 30, 12)
    trials_sample = st.slider("Сколько испытаний анализировать", 50, 500, 200, step=50)
    only_active = st.toggle("Только активные испытания", help="Идёт набор участников или испытание продолжается")
    with_trend = st.toggle("Динамика публикаций по годам", value=True,
                           help="Отдельный запрос в PubMed на каждый год — добавляет несколько секунд")
    st.divider()
    # Пароль защищает только нейросеть (чтобы чужие запросы не тратили дневной лимит ключа).
    # Без пароля поиск и статистика работают как обычно, а сводка собирается по шаблону.
    app_password = _secret("APP_PASSWORD")
    llm_allowed = llm.is_configured()
    if llm_allowed and app_password:
        if not st.session_state.get("llm_unlocked"):
            entered = st.text_input("Пароль для нейросети", type="password",
                                    help="Без пароля поиск и статистика тоже работают, "
                                         "но сводка собирается по шаблону.")
            if entered:
                if hmac.compare_digest(entered.encode(), app_password.encode()):
                    st.session_state["llm_unlocked"] = True
                    st.rerun()
                st.error("Неверный пароль")
        llm_allowed = st.session_state.get("llm_unlocked", False)
    if llm_allowed:
        st.success("Нейросеть подключена", icon=":material/check_circle:")
    elif llm.is_configured():
        st.info("Нейросеть выключена: введите пароль. Без него сводка собирается по шаблону, "
                "а свой запрос лучше писать на английском.", icon=":material/lock:")
    else:
        st.warning("Нейросеть не подключена — сводка будет собрана по шаблону. "
                   "Свой запрос лучше писать на английском.", icon=":material/info:")
    st.caption("Источники: [PubMed](https://pubmed.ncbi.nlm.nih.gov/) · "
               "[ClinicalTrials.gov](https://clinicaltrials.gov/)")

# ---------------------------------------------------------------- шапка и ввод
st.markdown(
    '<div class="hero-kicker">Мультиагентный поиск · психология и психиатрия</div>'
    '<div class="hero-title">PsyResearch</div>'
    '<div class="hero-sub">Шесть агентов ищут исследования в PubMed и клинические испытания '
    'в ClinicalTrials.gov, проверяют друг друга и собирают сводку со ссылками на источники.</div>',
    unsafe_allow_html=True,
)

with st.form("search", border=False):
    c_in, c_btn = st.columns([4, 1], vertical_alignment="bottom")
    c_in.text_input("Ваш вопрос", key="query",
                    placeholder="например: эффективность mindfulness при тревоге у студентов")
    if c_btn.form_submit_button("Найти", type="primary", icon=":material/search:", width="stretch"):
        st.session_state["run"] = True


def _pick_topic():
    name = st.session_state.get("topic_pill")
    if name:
        st.session_state["query"] = name
        st.session_state["run"] = True
        st.session_state["topic_pill"] = None


st.pills("Или готовая тема", list(PRESETS), key="topic_pill", on_change=_pick_topic)

# ---------------------------------------------------------------- запуск агентов
if st.session_state.pop("run", False) and st.session_state.get("query", "").strip():
    settings = orchestrator.Settings(
        max_articles=max_articles, trials_sample=trials_sample,
        year_from=year_from, year_to=year_to,
        only_active_trials=only_active, with_trend=with_trend, use_llm=llm_allowed,
    )
    with st.status("Агенты работают…", expanded=True) as status:
        result = orchestrator.run(
            st.session_state["query"].strip(), settings,
            on_step=lambda s: st.markdown(f"<span style='color:{MUTED}'>{s.seconds:>4} с</span> · "
                                          f"**{s.agent}** — {s.message}", unsafe_allow_html=True),
        )
        status.update(label=f"Готово за {result.trace[-1].seconds:.0f} с", state="complete", expanded=False)
    st.session_state["result"] = result

result: orchestrator.RunResult | None = st.session_state.get("result")
if result is None:
    st.markdown(
        '<div class="steps">'
        '<div class="step"><i>01</i><b>Планировщик</b><span>переводит вопрос в запросы на языке каждой базы</span></div>'
        '<div class="step"><i>02</i><b>Поиск и критик</b><span>два агента ищут параллельно, критик проверяет '
        'результаты и при необходимости возвращает запрос на доработку</span></div>'
        '<div class="step"><i>03</i><b>Методы → испытания</b><span>методы лечения из статей передаются агенту '
        'испытаний: что уже изучено и что проверяют сейчас</span></div>'
        '<div class="step"><i>04</i><b>Статистика и сводка</b><span>тренды, доли с доверительными интервалами '
        'и текст со ссылками, каждая из которых проверена</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ---------------------------------------------------------------- результаты
for source, err in result.errors.items():
    st.error(f"**{source}**: {err}")
if not result.plan.is_psychology:
    st.info("Похоже, запрос не про психологию — приложение всё равно поищет, но оно настроено на эту тему.")

f = result.stats.facts
st.markdown(f"### {result.plan.topic_ru}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Статей в PubMed", fmt(f.get("pubmed_total", 0)), help=f"За {year_from}–{year_to}")
m2.metric("Испытаний", fmt(f.get("trials_total", 0)), help="ClinicalTrials.gov")
m3.metric("Активных испытаний", f"{f.get('trials_active_share', 0)}%", help="Доля в проанализированной выборке")
if "behavioral_share" in f:
    b = f["behavioral_share"]
    m4.metric("С психотерапией", f"{b['pct']}%",
              help=f"Доля испытаний с поведенческим вмешательством, 95% ДИ {b['ci_low']}–{b['ci_high']}%, n={b['n']}")

tab_sum, tab_meth, tab_art, tab_tr, tab_stat, tab_agents = st.tabs(
    ["Сводка", "Методы", "Статьи", "Испытания", "Статистика", "Как работают агенты"]
)

with tab_sum:
    s = result.summary
    st.markdown(s.text)
    st.divider()
    cap = f"Сводку написал агент-аналитик ({s.model}). Ссылок на источники: {len(s.cited)}."
    if s.unknown_ids:
        st.warning("Аналитик сослался на источники, которых нет в найденных данных (возможная ошибка нейросети): "
                   + ", ".join(s.unknown_ids), icon=":material/warning:")
    elif s.cited:
        cap += " Все ссылки проверены — каждая есть среди найденных статей и испытаний."
    st.caption(cap)
    if s.error:
        with st.expander("Почему не сработала нейросеть"):
            st.code(s.error)

with tab_meth:
    mres = result.methods
    if not mres or not mres.methods:
        st.info("В найденных статьях не удалось выделить методы лечения.")
    else:
        st.markdown("#### Что пишут в статьях → что проверяют сейчас")
        st.caption(f"Агент PubMed выделил методы в найденных статьях ({mres.source}) и передал их агенту "
                   f"ClinicalTrials. Тот посчитал испытания по каждому методу, начатые с {year_from} года.")
        df_m = pd.DataFrame([{
            "Метод": m.name_ru, "Тип": m.kind, "Статей в выборке": len(m.pmids),
            "Испытаний всего": m.trials_total, "Идут сейчас": m.trials_active,
            "Статьи (PMID)": ", ".join(m.pmids[:6]),
        } for m in mres.methods])
        top = int(df_m["Испытаний всего"].fillna(0).max() or 1)
        st.dataframe(
            df_m, width="stretch", hide_index=True,
            column_config={
                "Испытаний всего": st.column_config.ProgressColumn(min_value=0, max_value=top, format="%d"),
                "Идут сейчас": st.column_config.NumberColumn(format="%d"),
            },
        )
        st.caption("Если метод часто встречается в статьях, но испытаний по нему мало, это может быть "
                   "пробелом в исследованиях — или признаком того, что метод уже хорошо изучен.")

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
        st.download_button("Скачать статьи (CSV)", df_art.to_csv(index=False).encode("utf-8-sig"),
                           "pubmed_articles.csv", "text/csv", icon=":material/download:")

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
            view, width="stretch", hide_index=True, height=480,
            column_config={
                "NCT": st.column_config.LinkColumn("NCT", display_text=r"study/(NCT\d+)"),
                "Участников": st.column_config.NumberColumn(format="%d"),
                "Старт": st.column_config.NumberColumn(format="%d"),
            },
        )
        st.download_button("Скачать испытания (CSV)", view.to_csv(index=False).encode("utf-8-sig"),
                           "clinical_trials.csv", "text/csv", icon=":material/download:")

with tab_stat:
    st_res = result.stats
    if not st_res.years.empty:
        st.markdown("#### Сколько публикаций выходит по теме каждый год")
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
      rankdir=LR; bgcolor="transparent"; nodesep=0.45; ranksep=0.55;
      node [shape=box, style="rounded,filled", fillcolor="#e9f2ec", color="#2f8f62", penwidth=1.2,
            fontname="Helvetica", fontcolor="#1c2a22", fontsize=12, margin="0.18,0.1"];
      edge [color="#9aaa9f", fontname="Helvetica", fontsize=10, fontcolor="#5d6c62", arrowsize=0.7];
      user [label="Вопрос", fillcolor="#ffffff", color="#c9d8cd"];
      planner [label="Планировщик\\n(нейросеть)"];
      pubmed [label="Агент PubMed"];
      trials [label="Агент\\nClinicalTrials"];
      critic [label="Критик\\n(нейросеть)"];
      stats [label="Статистик"];
      analyst [label="Аналитик\\n(нейросеть)"];
      ui [label="Сводка", fillcolor="#ffffff", color="#c9d8cd"];
      user -> planner; planner -> pubmed; planner -> trials;
      pubmed -> trials [label="методы\\nиз статей", color="#2f8f62", fontcolor="#23784f", constraint=false];
      pubmed -> critic; trials -> critic;
      critic -> planner [label="замечание →\\nповторный поиск", style=dashed, color="#2f8f62",
                         fontcolor="#23784f", constraint=false];
      critic -> stats; critic -> analyst; stats -> analyst; analyst -> ui;
    }
    """)
    st.markdown(
        "- **Планировщик** переводит вопрос на русском в запросы на языке каждого источника.\n"
        "- **Агенты PubMed и ClinicalTrials** ищут **параллельно**, каждый через своё API.\n"
        "- **Критик** оценивает, отвечают ли результаты на вопрос. Если нет — объясняет планировщику, "
        "что не так, и поиск повторяется (до двух раз). Остаётся лучшая попытка.\n"
        "- **Агент PubMed передаёт агенту ClinicalTrials** методы лечения из статей, "
        "и тот проверяет, сколько испытаний идёт по каждому.\n"
        "- **Статистик** считает динамику публикаций, распределения и доверительные интервалы.\n"
        "- **Аналитик** пишет сводку только по найденным данным; приложение проверяет, "
        "что каждая ссылка в сводке реально есть среди результатов."
    )
    if result.rounds:
        st.markdown("#### Проверки критика")
        st.dataframe(pd.DataFrame([{
            "Попытка": r.attempt,
            "Запрос PubMed": r.plan.pubmed_query,
            "Статей": r.pubmed_total, "Испытаний": r.trials_total,
            "Оценка": "—" if r.review.score is None else f"{r.review.score}/10",
            "Замечание критика": r.review.problem or r.review.model,
        } for r in result.rounds]), width="stretch", hide_index=True)
        if len(result.rounds) > 1:
            st.caption(f"Для результатов выбрана попытка {result.chosen_round} — с лучшей оценкой критика.")
    st.markdown("#### Журнал последнего запуска")
    st.dataframe(
        pd.DataFrame([{"Время, с": s.seconds, "Агент": s.agent, "Действие": s.message} for s in result.trace]),
        width="stretch", hide_index=True,
    )

st.divider()
st.caption("Учебный проект. Не является медицинской консультацией и не заменяет обращение к специалисту.")
