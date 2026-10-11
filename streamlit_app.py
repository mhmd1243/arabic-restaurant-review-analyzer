# streamlit_app.py
# The dashboard: reads reviews_analyzed.parquet and shows 4 tabs
# Run:
#   python process_dataset.py        # once
#   streamlit run streamlit_app.py   # never with plain python

import warnings
warnings.filterwarnings('ignore')
# Core imports
import os
import html
import json
import pandas as pd
import altair as alt
import streamlit as st
from core import ASPECT_AR, PRIORITY_AR, SENTIMENT_AR, ENTITY_COLORS
from core import BASE_SENTIMENT_MODEL, FINETUNED_DIR
from core import load_models, analyze, summarize, split_of

DATA_FILE = "reviews_analyzed.parquet"
FINETUNE_RESULTS = os.path.join(FINETUNED_DIR, "eval_results.json")

# Colors: every color has one job
# sentiment = two opposite hues (orange / blue) + gray for neutral
SENT_COLORS = {"negative": "#d95926", "positive": "#3987e5", "neutral": "#8f8e88"}
# priority = status colors, always shown next to the word
PRIO_COLORS = {"high": "#d03b3b", "medium": "#fab219", "low": "#0ca30c"}
ACCENT = "#3987e5"
TEXT = "#f0efec"    # main text
TEXT_2 = "#c3c2b7"  # labels
TEXT_3 = "#8f8e88"  # small notes
CARD = "#232321"    # card background
LINE = "#33332f"    # borders
GRID = "#2c2c2a"    # chart grid
FONT = "IBM Plex Sans Arabic"

st.set_page_config(page_title="تحليل ريفيوهات المطاعم", page_icon="🍽️", layout="wide")

# Right-to-left only on text, not on the whole page
# (flipping the whole page broke Streamlit's own layout: tabs, sliders, tables)
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');

    [data-testid="stMarkdownContainer"], [data-testid="stHeading"], [data-testid="stCaptionContainer"],
    [data-testid="stWidgetLabel"], [data-testid="stAlert"], .stTextArea textarea,
    .stTabs button p, h1, h2, h3 {{
        font-family: '{FONT}', sans-serif;
    }}
    [data-testid="stMarkdownContainer"], [data-testid="stHeading"], [data-testid="stCaptionContainer"],
    [data-testid="stWidgetLabel"], [data-testid="stAlert"], .stTextArea textarea {{
        direction: rtl; text-align: right;
    }}
    .block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
    h1 {{ font-weight: 700; letter-spacing: 0; }}
    h3 {{ font-weight: 600; margin-top: .6rem; }}

    .kpi-row {{ display: flex; flex-wrap: wrap; gap: 12px; direction: rtl; margin: 6px 0 22px; }}
    .kpi {{ flex: 1 1 170px; background: {CARD}; border: 1px solid {LINE}; border-radius: 12px;
            padding: 14px 18px; }}
    .kpi .label {{ color: {TEXT_2}; font-size: .86rem; }}
    .kpi .value {{ color: {TEXT}; font-size: 1.75rem; font-weight: 600; line-height: 1.35;
                   font-variant-numeric: tabular-nums; }}
    .kpi .sub {{ color: {TEXT_3}; font-size: .8rem; }}

    .badge {{ display: inline-flex; align-items: center; gap: 6px; font-weight: 600; }}
    .badge .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}

    .card {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 12px; padding: 14px 18px;
             margin: 4px 0 16px; line-height: 1.9; }}
    .entity {{ padding: 2px 7px; border-radius: 6px; margin: 0 2px; color: #111; }}
    .entity small {{ font-size: .72em; opacity: .75; margin-right: 4px; }}
    .muted {{ color: {TEXT_3}; font-size: .85rem; }}
    .reasons ul {{ margin: 0; padding: 0 1.2rem 0 0; text-align: right; }}
    .reasons li {{ margin: 2px 0; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------
@st.cache_data
def get_data():
    if not os.path.exists(DATA_FILE):
        return None
    df = pd.read_parquet(DATA_FILE)
    df["review_len"] = df["text"].str.len()
    if "split" not in df:  # file from an older version or from the notebook
        df["split"] = df["text"].map(split_of)
    if "sentiment_model" not in df:
        df["sentiment_model"] = BASE_SENTIMENT_MODEL
    return df


def load_finetune_results():
    if not os.path.exists(FINETUNE_RESULTS):
        return None
    with open(FINETUNE_RESULTS, encoding="utf-8") as f:
        return json.load(f)


def macro_f1(true, pred):
    # average F1 of both classes (fairer than accuracy: ~70% of reviews are positive)
    scores = []
    for c in ["negative", "positive"]:
        tp = ((pred == c) & (true == c)).sum()
        fp = ((pred == c) & (true != c)).sum()
        fn = ((pred != c) & (true == c)).sum()
        if tp == 0:
            scores.append(0.0)
        else:
            scores.append(2 * tp / (2 * tp + fp + fn))
    return sum(scores) / 2


# models are loaded once per server (cache_resource), only when needed
@st.cache_resource(show_spinner="بنحمّل الموديلات... أول مرة بتاخد وقت")
def get_models():
    return load_models(with_summarizer=True)


# ------------------------------------------------------------------
# UI pieces
# ------------------------------------------------------------------
def kpi_row(items):
    # row of number cards. items = [(label, value), or (label, value, small note)]
    cards = []
    for item in items:
        label = item[0]
        value = item[1]
        sub_html = ""
        if len(item) > 2 and item[2]:
            sub_html = f'<div class="sub">{item[2]}</div>'
        cards.append(f'<div class="kpi"><div class="label">{label}</div>'
                     f'<div class="value">{value}</div>{sub_html}</div>')
    st.markdown(f'<div class="kpi-row">{"".join(cards)}</div>', unsafe_allow_html=True)


def badge(text, color):
    return f'<span class="badge"><span class="dot" style="background:{color}"></span>{text}</span>'


def highlight_entities(text, entities):
    # color every entity inside the text. html.escape stops a review from injecting HTML
    parts = []
    last = 0
    for e in sorted(entities, key=lambda x: x["start"]):
        parts.append(html.escape(text[last:e["start"]]))
        color = ENTITY_COLORS.get(e["label"], "#e5e7eb")
        parts.append(f'<span class="entity" style="background:{color}">'
                     f'{html.escape(text[e["start"]:e["end"]])}<small>{e["label"]}</small></span>')
        last = e["end"]
    parts.append(html.escape(text[last:]))
    return f'<div class="card">{"".join(parts)}</div>'


def style_chart(chart, height):
    # same look for every chart: transparent background, quiet grid, Arabic font
    return (chart.properties(height=height, background="transparent")
                 .configure_view(strokeWidth=0)
                 .configure_axis(labelColor=TEXT_2, titleColor=TEXT_3, gridColor=GRID,
                                 domainColor=LINE, tickColor=LINE, labelFont=FONT, titleFont=FONT,
                                 labelFontSize=12, titleFontSize=11, titleFontWeight="normal")
                 .configure_scale(bandPaddingInner=0.35)
                 .configure_legend(labelColor=TEXT_2, labelFont=FONT, labelFontSize=12,
                                   orient="top", symbolType="circle", direction="horizontal",
                                   columnPadding=18, labelOffset=6)
                 .configure_text(font=FONT))


def aspect_counts(frame):
    # how many times each aspect is mentioned, per sentiment
    # "food|price" -> 2 rows (a review can mention more than one aspect)
    d = frame[["aspects", "pred_sentiment"]].copy()
    d["aspect"] = d["aspects"].str.split("|")
    d = d.explode("aspect").reset_index(drop=True)  # reset_index: explode repeats the index
    d = d[d["aspect"] != "general"]
    return d.groupby(["aspect", "pred_sentiment"]).size().rename("n").reset_index()


def mentions_chart(frame, height=300):
    # horizontal bars: mentions of each aspect in negative vs positive reviews
    # (horizontal so the Arabic labels are readable; x reversed so bars grow right to left)
    d = aspect_counts(frame)
    d = d[d["pred_sentiment"].isin(["negative", "positive"])].copy()
    if d.empty:
        return None
    d["aspect_ar"] = d["aspect"].map(ASPECT_AR)
    d["sentiment_ar"] = d["pred_sentiment"].map(SENTIMENT_AR)
    order = d.groupby("aspect_ar")["n"].sum().sort_values(ascending=False).index.tolist()
    sent_order = [SENTIMENT_AR["negative"], SENTIMENT_AR["positive"]]
    chart = alt.Chart(d).mark_bar(cornerRadiusEnd=4, height={"band": 1}).encode(
        y=alt.Y("aspect_ar:N", sort=order, title=None,
                axis=alt.Axis(orient="right", labelLimit=220, ticks=False, domain=False)),
        x=alt.X("n:Q", title="عدد مرات الذكر", scale=alt.Scale(reverse=True),
                axis=alt.Axis(tickCount=4, format="~s")),
        yOffset=alt.YOffset("sentiment_ar:N", sort=sent_order, scale=alt.Scale(paddingInner=0.15)),
        color=alt.Color("sentiment_ar:N", sort=sent_order, title=None,
                        scale=alt.Scale(domain=sent_order,
                                        range=[SENT_COLORS["negative"], SENT_COLORS["positive"]])),
        tooltip=[alt.Tooltip("aspect_ar:N", title="الجانب"),
                 alt.Tooltip("sentiment_ar:N", title="المشاعر"),
                 alt.Tooltip("n:Q", title="عدد مرات الذكر", format=",")],
    )
    return style_chart(chart, height)


def complaint_rate_chart(frame, height=300):
    # complaint rate: of all mentions of an aspect, the share that was in negative reviews
    d = aspect_counts(frame)
    if d.empty:
        return None
    p = d.pivot_table(index="aspect", columns="pred_sentiment", values="n", fill_value=0)
    total = p.sum(axis=1)
    rate = (p.get("negative", 0) / total).rename("rate").reset_index()
    rate["total"] = total.values
    rate["aspect_ar"] = rate["aspect"].map(ASPECT_AR)
    rate["label"] = (rate["rate"] * 100).round().astype(int).astype(str) + "%"
    order = rate.sort_values("rate", ascending=False)["aspect_ar"].tolist()
    top = max(rate["rate"].max() * 1.2, 0.1)
    base = alt.Chart(rate).encode(
        y=alt.Y("aspect_ar:N", sort=order, title=None,
                axis=alt.Axis(orient="right", labelLimit=220, ticks=False, domain=False)),
        x=alt.X("rate:Q", title="نسبة الذكر في ريفيوهات سلبية",
                scale=alt.Scale(domain=[0, top], reverse=True), axis=alt.Axis(format="%", tickCount=4)),
        tooltip=[alt.Tooltip("aspect_ar:N", title="الجانب"),
                 alt.Tooltip("rate:Q", title="نسبة الشكوى", format=".0%"),
                 alt.Tooltip("total:Q", title="مرات الذكر", format=",")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height={"band": 0.6}, color=SENT_COLORS["negative"])
    labels = base.mark_text(align="right", dx=-6, color=TEXT_2, fontSize=12).encode(text="label:N")
    return style_chart(bars + labels, height)


# ------------------------------------------------------------------
# Page
# ------------------------------------------------------------------
df = get_data()

st.title("تحليل ريفيوهات المطاعم")
if df is not None:
    st.caption(f"تحليل المشاعر واستخراج الكيانات والتلخيص على {len(df):,} ريفيو حقيقي")
else:
    st.caption("تحليل المشاعر واستخراج الكيانات والتلخيص")

if df is None:
    st.error("مفيش داتا متحللة لسه. شغّل الأمر ده الأول:\n\n`python process_dataset.py`")

tab_overview, tab_restaurant, tab_live, tab_eval = st.tabs(
    ["نظرة عامة", "مطعم معين", "حلّل ريفيو جديد", "تقييم الموديل"])

# ------------------------------------------------------------------
# Tab 1: Overview
# ------------------------------------------------------------------
with tab_overview:
    if df is not None:
        kpi_row([
            ("عدد الريفيوهات", f"{len(df):,}"),
            ("عدد المطاعم", f"{df['restaurant_id'].nunique():,}"),
            ("نسبة الريفيوهات السلبية", f"{(df['pred_sentiment'] == 'negative').mean():.0%}"),
            ("ريفيوهات أولويتها عالية", f"{(df['priority'] == 'high').sum():,}", "محتاجة رد سريع"),
        ])

        left, right = st.columns(2, gap="large")
        with right:
            st.subheader("الجوانب الأكتر مشاكل")
            st.caption("من كل مرة اتذكر فيها الجانب، كام في المية كانت في ريفيو سلبي")
            c = complaint_rate_chart(df)
            if c is not None:
                st.altair_chart(c, width="stretch")
        with left:
            st.subheader("الناس بتتكلم عن إيه")
            st.caption("عدد مرات ذكر كل جانب في الريفيوهات السلبية والإيجابية")
            c = mentions_chart(df)
            if c is not None:
                st.altair_chart(c, width="stretch")

        st.subheader("المطاعم اللي محتاجة تدخّل")
        min_reviews = st.slider("أقل عدد ريفيوهات للمطعم (عشان النسب تبقى منطقية)", 5, 100, 20)
        # one row per restaurant
        temp = df.copy()
        temp["is_negative"] = (temp["pred_sentiment"] == "negative") * 100
        temp["is_high"] = (temp["priority"] == "high").astype(int)
        stats = temp.groupby("restaurant_id").agg(
            reviews=("text", "size"),
            negative_pct=("is_negative", "mean"),
            high_priority=("is_high", "sum"),
            top_complaint=("main_aspect", lambda s: s.mode().iat[0])
        )
        # restaurants with too few reviews give misleading percentages (1 bad review = 100%)
        stats = stats[stats["reviews"] >= min_reviews]
        stats = stats.sort_values(["negative_pct", "high_priority"], ascending=False)
        stats = stats.head(15).reset_index()
        stats["top_complaint"] = stats["top_complaint"].map(ASPECT_AR)
        st.dataframe(
            stats,
            column_config={
                "restaurant_id": st.column_config.TextColumn("المطعم"),
                "reviews": st.column_config.NumberColumn("عدد الريفيوهات"),
                "negative_pct": st.column_config.ProgressColumn(
                    "نسبة السلبي", format="%.0f%%", min_value=0, max_value=100),
                "high_priority": st.column_config.NumberColumn("أولوية عالية"),
                "top_complaint": st.column_config.TextColumn("أكتر جانب متذكر"),
            },
            width="stretch", hide_index=True,
        )

# ------------------------------------------------------------------
# Tab 2: One restaurant
# ------------------------------------------------------------------
with tab_restaurant:
    if df is not None:
        counts = df["restaurant_id"].value_counts()
        rest_id = st.selectbox("اختار المطعم (مرتبين بعدد الريفيوهات)", counts.index,
                               format_func=lambda r: f"مطعم {r}  ·  {counts[r]} ريفيو")
        rdf = df[df["restaurant_id"] == rest_id]

        kpi_row([
            ("عدد الريفيوهات", f"{len(rdf):,}"),
            ("نسبة الإيجابي", f"{(rdf['pred_sentiment'] == 'positive').mean():.0%}"),
            ("نسبة السلبي", f"{(rdf['pred_sentiment'] == 'negative').mean():.0%}"),
            ("أولوية عالية", f"{int((rdf['priority'] == 'high').sum())}"),
        ])

        st.subheader("نقط القوة والضعف")
        c = mentions_chart(rdf, height=220)
        if c is not None:
            st.altair_chart(c, width="stretch")
        else:
            st.info("الريفيوهات دي مفيهاش جوانب واضحة")

        st.subheader("أهم الريفيوهات اللي محتاجة رد")
        urgent = rdf.sort_values(["priority_score", "confidence"], ascending=False).head(10)
        urgent = urgent[["text", "priority", "reasons", "entities"]].copy()
        urgent["priority"] = urgent["priority"].map(PRIORITY_AR)
        st.dataframe(
            urgent,
            column_config={
                "text": st.column_config.TextColumn("الريفيو", width="large"),
                "priority": st.column_config.TextColumn("الأولوية", width="small"),
                "reasons": st.column_config.TextColumn("ليه", width="medium"),
                "entities": st.column_config.TextColumn("الكيانات", width="small"),
            },
            width="stretch", hide_index=True,
        )

        st.subheader("لخّص ريفيو")
        longest = rdf.sort_values("review_len", ascending=False)
        choice = st.selectbox("اختار ريفيو (الأطول الأول)", longest.index,
                              format_func=lambda i: f"{longest.at[i, 'review_len']} حرف  ·  "
                                                    f"{longest.at[i, 'text'][:60]}...")
        if st.button("لخّص", type="primary"):
            with st.spinner("بنلخّص..."):
                summary = summarize(df.at[choice, "text"], get_models())
            st.markdown(f'<div class="card">{html.escape(summary)}</div>', unsafe_allow_html=True)
            with st.expander("الريفيو الأصلي"):
                st.write(df.at[choice, "text"])

# ------------------------------------------------------------------
# Tab 3: Analyze a new review
# ------------------------------------------------------------------
with tab_live:
    text = st.text_area("اكتب ريفيو", height=130,
                        placeholder="مثلًا: الأكل كان بارد والخدمة بطيئة جدًا والأسعار مبالغ فيها")
    if st.button("حلّل", type="primary"):
        if not text.strip():
            st.warning("اكتب الريفيو الأول")
        else:
            with st.spinner("بنحلّل..."):
                r = analyze(text, get_models())
            kpi_row([
                ("المشاعر", badge(SENTIMENT_AR.get(r["sentiment"], r["sentiment"]),
                                  SENT_COLORS.get(r["sentiment"], TEXT_3)),
                 f"ثقة الموديل {r['confidence']:.0%}"),
                ("الأولوية", badge(PRIORITY_AR[r["priority"]], PRIO_COLORS[r["priority"]]),
                 f"{r['priority_score']} نقاط"),
                ("الجوانب", "، ".join(ASPECT_AR[a] for a in r["aspects"])),
            ])
            st.subheader("الملخص")
            st.markdown(f'<div class="card">{html.escape(r["summary"])}</div>', unsafe_allow_html=True)

            st.subheader("ليه الأولوية دي؟")
            # the list is built as HTML with dir="rtl" so the reasons start from the right
            items = ""
            for reason in r["reasons"]:
                items += f"<li>{html.escape(reason)}</li>"
            st.markdown(f'<div class="card reasons" dir="rtl"><ul>{items}</ul></div>',
                        unsafe_allow_html=True)

# ------------------------------------------------------------------
# Tab 4: Model evaluation (test split only)
# ------------------------------------------------------------------
with tab_eval:
    if df is not None:
        model_used = df["sentiment_model"].iloc[0]
        is_finetuned = model_used != BASE_SENTIMENT_MODEL
        ev = df[df["split"] == "test"]
        st.markdown(
            f"التقييم على **{len(ev):,} ريفيو test** بس (20% من الداتا ثابتة). "
            "الموديل المتدرب ماشافهاش وقت التدريب، فالأرقام دي صادقة.")
        st.caption(f"الموديل المستخدم: {'بعد الـ fine-tuning' if is_finetuned else 'الأصلي من غير تدريب'}"
                   f"  ·  {model_used}")

        correct = ev["pred_sentiment"] == ev["true_sentiment"]
        committed = ev["pred_sentiment"] != "neutral"
        majority = ev["true_sentiment"].value_counts(normalize=True).iloc[0]
        has_neutral = (~committed).any()

        cards = [("الدقة", f"{correct.mean():.1%}", "أي محايد بيتحسب غلط" if has_neutral else "من كل ريفيوهات الـ test"),
                 ("Macro F1", f"{macro_f1(ev['true_sentiment'], ev['pred_sentiment']):.3f}", "متوسط الفئتين، أعدل مع الداتا غير المتوازنة")]
        if has_neutral:
            cards.append(("قال محايد", f"{(~committed).mean():.1%}", f"الدقة من غيرهم {correct[committed].mean():.1%}"))
        cards.append(("خط الأساس", f"{majority:.1%}", "لو خمّنا الفئة الأكبر دايمًا"))
        kpi_row(cards)

        results = load_finetune_results()
        if results:
            st.subheader("قبل وبعد الـ fine-tuning")
            st.caption(f"نفس الـ {results['test_size']:,} ريفيو test للموديلين")
            rows = []
            for key, name in (("base", "الموديل الأصلي"), ("finetuned", "بعد الـ fine-tuning")):
                rows.append({"model": name, "metric": "الدقة", "value": results[key]["accuracy"]})
                rows.append({"model": name, "metric": "Macro F1", "value": results[key]["f1_macro"]})
            comp = pd.DataFrame(rows)
            comp["label"] = comp.apply(lambda r: f"{r['value']:.1%}" if r["metric"] == "الدقة"
                                       else f"{r['value']:.3f}", axis=1)
            model_order = ["الموديل الأصلي", "بعد الـ fine-tuning"]
            cbase = alt.Chart(comp).encode(
                x=alt.X("metric:N", title=None, axis=alt.Axis(labelAngle=0, ticks=False)),
                xOffset=alt.XOffset("model:N", sort=model_order, scale=alt.Scale(paddingInner=0.15)),
                y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1.12]),
                        axis=alt.Axis(format=".0%", tickCount=5)),
                tooltip=[alt.Tooltip("model:N", title="الموديل"), alt.Tooltip("metric:N", title="المقياس"),
                         alt.Tooltip("value:Q", title="القيمة", format=".3f")],
            )
            cbars = cbase.mark_bar(cornerRadiusEnd=4).encode(
                color=alt.Color("model:N", sort=model_order, title=None,
                                scale=alt.Scale(domain=model_order, range=[TEXT_3, ACCENT])))
            clabels = cbase.mark_text(dy=-9, color=TEXT_2, fontSize=12).encode(text="label:N")
            st.altair_chart(style_chart(cbars + clabels, 260), width="stretch")

        left, right = st.columns(2, gap="large")
        with right:
            st.subheader("Confusion Matrix")
            st.caption("الصفوف = التقييم الحقيقي، الأعمدة = توقع الموديل")
            cm = ev.groupby(["true_sentiment", "pred_sentiment"]).size().rename("n").reset_index()
            cm["row_pct"] = cm["n"] / cm.groupby("true_sentiment")["n"].transform("sum")
            cm["true_ar"] = cm["true_sentiment"].map(SENTIMENT_AR)
            cm["pred_ar"] = cm["pred_sentiment"].map(SENTIMENT_AR)
            cm["label"] = cm["n"].map("{:,}".format) + "\n" + (cm["row_pct"] * 100).round().astype(int).astype(str) + "%"
            pred_keys = ("negative", "neutral", "positive") if has_neutral else ("negative", "positive")
            pred_order = [SENTIMENT_AR[k] for k in pred_keys]
            true_order = [SENTIMENT_AR[k] for k in ("negative", "positive")]
            base = alt.Chart(cm).encode(
                x=alt.X("pred_ar:N", sort=pred_order, title="توقع الموديل",
                        axis=alt.Axis(orient="top", labelAngle=0, ticks=False, domain=False)),
                y=alt.Y("true_ar:N", sort=true_order, title="الحقيقي",
                        axis=alt.Axis(orient="right", ticks=False, domain=False)),
            )
            heat = base.mark_rect(cornerRadius=6, stroke="#1a1a19", strokeWidth=3).encode(
                color=alt.Color("row_pct:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]), legend=None),
                tooltip=[alt.Tooltip("true_ar:N", title="الحقيقي"), alt.Tooltip("pred_ar:N", title="التوقع"),
                         alt.Tooltip("n:Q", title="العدد", format=","),
                         alt.Tooltip("row_pct:Q", title="من الصف", format=".0%")],
            )
            txt = base.mark_text(fontSize=13, lineBreak="\n").encode(
                text="label:N",
                color=alt.condition("datum.row_pct > 0.5", alt.value("#ffffff"), alt.value("#1a1a19")),
            )
            st.altair_chart(style_chart(heat + txt, 220), width="stretch")

        with left:
            st.subheader("هل ثقة الموديل بتعني إنه صح؟")
            st.caption("الدقة لكل مستوى ثقة. المفروض تزيد كل ما الثقة تزيد")
            sub = ev[committed].copy()
            sub["bucket"] = pd.cut(sub["confidence"], [0, 0.6, 0.7, 0.8, 0.9, 1.0],
                                   labels=["< 60%", "60-70%", "70-80%", "80-90%", "90%+"])
            calib = (sub.assign(ok=sub["pred_sentiment"] == sub["true_sentiment"])
                        .groupby("bucket", observed=True)
                        .agg(acc=("ok", "mean"), n=("ok", "size")).reset_index())
            calib["bucket"] = calib["bucket"].astype(str)
            calib["label"] = (calib["acc"] * 100).round().astype(int).astype(str) + "%"
            cbase = alt.Chart(calib).encode(
                x=alt.X("bucket:N", sort=None, title="ثقة الموديل", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("acc:Q", title=None, scale=alt.Scale(domain=[0, 1.1]),
                        axis=alt.Axis(format="%", tickCount=5)),
                tooltip=[alt.Tooltip("bucket:N", title="الثقة"),
                         alt.Tooltip("acc:Q", title="الدقة", format=".1%"),
                         alt.Tooltip("n:Q", title="عدد الريفيوهات", format=",")],
            )
            cbars = cbase.mark_bar(cornerRadiusEnd=4, width={"band": 0.6}, color=ACCENT)
            clabels = cbase.mark_text(dy=-9, color=TEXT_2, fontSize=12).encode(text="label:N")
            st.altair_chart(style_chart(cbars + clabels, 220), width="stretch")

        st.subheader("أخطاء الموديل وهو واثق من نفسه")
        st.caption("أهم جزء في التحليل: كتير منها الـ label اللي في الداتا هو الغلط مش الموديل "
                   "(label noise)، والباقي سخرية أو ريفيو مختلط")
        wrong = ev[~correct & committed].sort_values("confidence", ascending=False).head(20)
        wrong = wrong[["text", "true_sentiment", "pred_sentiment", "confidence"]].copy()
        wrong["true_sentiment"] = wrong["true_sentiment"].map(SENTIMENT_AR)
        wrong["pred_sentiment"] = wrong["pred_sentiment"].map(SENTIMENT_AR)
        st.dataframe(
            wrong,
            column_config={
                "text": st.column_config.TextColumn("الريفيو", width="large"),
                "true_sentiment": st.column_config.TextColumn("label الداتا", width="small"),
                "pred_sentiment": st.column_config.TextColumn("توقع الموديل", width="small"),
                "confidence": st.column_config.ProgressColumn("الثقة", format="%.2f",
                                                              min_value=0, max_value=1, width="small"),
            },
            width="stretch", hide_index=True,
        )
