import pandas as pd
import plotly.express as px
import streamlit as st

DATA_URL = "https://raw.githubusercontent.com/sbin058/data/refs/heads/main/stroke.csv"

st.set_page_config(page_title="탐색 - 뇌졸중 예측 실습실", page_icon="🔍", layout="wide")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🔍 탐색")
st.caption("뇌졸중 예측 실습실 · 데이터를 그림과 표로 살펴봐요.")

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

# 그림에서 쓸 이름표
STROKE_LABEL = {0: "겪지 않음", 1: "겪음"}
df = df.copy()
df["뇌졸중"] = df["stroke"].map(STROKE_LABEL)
stroke_order = ["겪지 않음", "겪음"]

# ---------- 1. 히스토그램 ----------
st.header("1. 나이와 평균 혈당의 분포")
h1, h2 = st.columns(2)
with h1:
    fig = px.histogram(
        df, x="age", nbins=30,
        labels={"age": "나이", "count": "사람 수"},
        title="나이 분포",
    )
    fig.update_layout(yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)
with h2:
    fig = px.histogram(
        df, x="avg_glucose_level", nbins=40,
        labels={"avg_glucose_level": "평균 혈당"},
        title="평균 혈당 분포",
    )
    fig.update_layout(yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)

# ---------- 2. 상자그림 ----------
st.header("2. 뇌졸중 여부에 따른 나이·평균 혈당 비교")
b1, b2 = st.columns(2)
with b1:
    fig = px.box(
        df, x="뇌졸중", y="age", color="뇌졸중",
        category_orders={"뇌졸중": stroke_order},
        labels={"age": "나이", "뇌졸중": "뇌졸중 경험"},
        title="나이",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with b2:
    fig = px.box(
        df, x="뇌졸중", y="avg_glucose_level", color="뇌졸중",
        category_orders={"뇌졸중": stroke_order},
        labels={"avg_glucose_level": "평균 혈당", "뇌졸중": "뇌졸중 경험"},
        title="평균 혈당",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("두 그룹의 평균값")
mean_table = (
    df.groupby("뇌졸중")[["age", "avg_glucose_level"]]
    .mean()
    .reindex(stroke_order)
    .round(2)
    .rename(columns={"age": "나이 평균", "avg_glucose_level": "평균 혈당의 평균"})
    .reset_index()
    .rename(columns={"뇌졸중": "뇌졸중 경험"})
)
st.dataframe(mean_table, hide_index=True, use_container_width=True)

# ---------- 3. 고혈압·심장병과 뇌졸중 비율 ----------
st.header("3. 고혈압·심장병 여부에 따른 뇌졸중 비율")


def rate_chart(column: str, title: str):
    rate = (
        df.groupby(column)["stroke"].mean().mul(100).round(2).reset_index()
    )
    rate["구분"] = rate[column].map({0: "없음", 1: "있음"})
    rate = rate.rename(columns={"stroke": "뇌졸중 비율(%)"})
    fig = px.bar(
        rate, x="구분", y="뇌졸중 비율(%)", color="구분",
        category_orders={"구분": ["없음", "있음"]},
        text="뇌졸중 비율(%)",
        title=title,
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(showlegend=False, xaxis_title=title.split("에")[0])
    fig.update_yaxes(range=[0, max(rate["뇌졸중 비율(%)"].max() * 1.25, 1)])
    return fig


r1, r2 = st.columns(2)
with r1:
    st.plotly_chart(rate_chart("hypertension", "고혈압 여부에 따른 뇌졸중 비율"), use_container_width=True)
with r2:
    st.plotly_chart(rate_chart("heart_disease", "심장병 여부에 따른 뇌졸중 비율"), use_container_width=True)

# ---------- 4. bmi가 빈 사람들 ----------
st.header("4. 체질량지수(bmi)가 비어 있는 사람들")
missing = df[df["bmi"].isna()]
rows = []
for name, part in [("bmi가 빈 사람들", missing), ("전체", df)]:
    n = len(part)
    n_stroke = int((part["stroke"] == 1).sum())
    rows.append(
        {
            "구분": name,
            "사람 수": n,
            "뇌졸중 1인 사람 수": n_stroke,
            "뇌졸중 비율(%)": round(n_stroke / n * 100, 2) if n else 0.0,
        }
    )
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ---------- 5. 흡연 상태 ----------
st.header("5. 흡연 상태별 사람 수")
smoke = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
st.dataframe(smoke, hide_index=True, use_container_width=True)
