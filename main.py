import json
from pathlib import Path

import pandas as pd
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"
APP_ICON = "🧠"
DATA_URL = "https://raw.githubusercontent.com/sbin058/data/refs/heads/main/stroke.csv"

BASE_DIR = Path(__file__).parent
SAVE_DIR = BASE_DIR / "내_기록"
SAVE_FILE = SAVE_DIR / "기록.json"

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

# 큰 숫자 카드 스타일
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 2.8rem; font-weight: 700; }
    [data-testid="stMetricLabel"] p { font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- 저장 / 불러오기 ----------
def load_record():
    """내_기록/기록.json 에서 우리말 뜻과 출처를 불러온다."""
    if SAVE_FILE.exists():
        try:
            data = json.loads(SAVE_FILE.read_text(encoding="utf-8"))
            return {
                "meanings": dict(data.get("meanings", {})),
                "source": str(data.get("source", "")),
            }
        except (json.JSONDecodeError, OSError):
            pass
    return {"meanings": {}, "source": ""}


def save_record(meanings, source):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"meanings": meanings, "source": source}
    SAVE_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------- 데이터 ----------
@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def describe_values(series: pd.Series) -> str:
    """열의 '값의 종류'를 짧은 글로 만든다."""
    values = series.dropna().unique()
    if len(values) <= 6:
        shown = sorted(values.tolist(), key=lambda v: str(v))
        return ", ".join(str(v) for v in shown)
    if pd.api.types.is_numeric_dtype(series):
        return f"숫자 ({series.min():g} ~ {series.max():g})"
    return f"글자 ({len(values)}가지)"


def build_table(df: pd.DataFrame, meanings: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "열 이름": df.columns,
            "우리말 뜻": [meanings.get(c, "") for c in df.columns],
            "값의 종류": [describe_values(df[c]) for c in df.columns],
            "빈 값 개수": [int(df[c].isna().sum()) for c in df.columns],
        }
    )


# ---------- 화면 ----------
st.title(f"{APP_ICON} {APP_TITLE}")

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

# 세션 처음 한 번만 저장된 기록을 불러온다 (새로고침해도 이어서 보이도록)
if "loaded" not in st.session_state:
    record = load_record()
    st.session_state["base_table"] = build_table(df, record["meanings"])
    st.session_state["saved_meanings"] = record["meanings"]
    st.session_state["source_text"] = record["source"]
    st.session_state["saved_source"] = record["source"]
    st.session_state["loaded"] = True

st.header("1. 이 데이터는 무엇일까요?")

# 큰 숫자 카드 네 개
total = len(df)
n_cols = df.shape[1]
n_stroke = int((df["stroke"] == 1).sum())
ratio = n_stroke / total * 100 if total else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    with st.container(border=True):
        st.metric("전체 사람 수", f"{total:,}명")
with c2:
    with st.container(border=True):
        st.metric("열 개수", f"{n_cols}개")
with c3:
    with st.container(border=True):
        st.metric("stroke = 1인 사람 수", f"{n_stroke:,}명")
with c4:
    with st.container(border=True):
        st.metric("stroke = 1 비율", f"{ratio:.2f}%")

# 열 설명 표 (우리말 뜻만 수정 가능)
st.subheader("열 설명 표")
st.caption("‘우리말 뜻’ 칸을 눌러 교재를 보고 직접 적어 보세요. 적은 내용은 자동으로 저장돼요.")

edited = st.data_editor(
    st.session_state["base_table"],
    key="column_table",
    hide_index=True,
    use_container_width=True,
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름"),
        "우리말 뜻": st.column_config.TextColumn("우리말 뜻", help="직접 적어 넣는 칸"),
        "값의 종류": st.column_config.TextColumn("값의 종류"),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수"),
    },
)

# 처음 다섯 줄
st.subheader("데이터 처음 다섯 줄")
st.dataframe(df.head(5), use_container_width=True, hide_index=True)

# 데이터 출처
st.subheader("데이터 출처")
st.text_area(
    "출처를 적어 주세요 (교재에 나온 내용)",
    key="source_text",
    height=120,
    placeholder="여기에 교재에 적힌 출처를 적어 주세요.",
)

# ---------- 변경되면 저장 ----------
new_meanings = {
    row["열 이름"]: ("" if pd.isna(row["우리말 뜻"]) else str(row["우리말 뜻"]))
    for _, row in edited.iterrows()
}
new_meanings = {k: v for k, v in new_meanings.items() if v.strip()}
new_source = st.session_state["source_text"]

if (
    new_meanings != st.session_state["saved_meanings"]
    or new_source != st.session_state["saved_source"]
):
    try:
        save_record(new_meanings, new_source)
        st.session_state["saved_meanings"] = new_meanings
        st.session_state["saved_source"] = new_source
        st.toast("내_기록 폴더에 저장했어요 ✅")
    except OSError as e:
        st.error(f"저장하지 못했어요: {e}")
