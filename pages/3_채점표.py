import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_URL = "https://raw.githubusercontent.com/sbin058/data/refs/heads/main/stroke.csv"

st.set_page_config(page_title="채점표 - 뇌졸중 예측 실습실", page_icon="📝", layout="wide")

KOREAN = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
ALL_FEATURES = list(KOREAN)
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

LR_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "기준 모델(훈련용에서 많은 쪽으로만 답하는 모델)"

NA = "계산할 수 없음"
MAX_ROWS = 20


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("📝 채점표")
st.caption("뇌졸중 예측 실습실 · 테스트 데이터로 모델의 답을 하나하나 채점해 봐요. 뇌졸중을 양성으로 둬요.")

try:
    raw = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

# ---------- 입력 속성 고르기 ('분류 모델' 페이지와 같은 방법) ----------
st.header("1. 입력으로 사용할 속성")
chosen = st.multiselect(
    "입력 속성 (두 개 이상)",
    options=ALL_FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda c: KOREAN[c],
)
selected = [c for c in ALL_FEATURES if c in chosen]
if len(selected) < 2:
    st.info("입력 속성을 두 개 이상 골라 주세요. 두 개 이상 고르면 모델을 만들어 보여 드려요.")
    st.stop()

# ---------- 데이터 나누기 ('분류 모델' 페이지와 같은 방법) ----------
df = raw.sort_values("id", kind="mergesort").reset_index(drop=True)
is_test = (df.index % 10) < 3  # 열 명씩 묶어 앞 세 명은 테스트용
X_all = df[selected].copy()
y_all = df["stroke"]
X_train, X_test = X_all[~is_test], X_all[is_test]
y_train, y_test = y_all[~is_test], y_all[is_test]

if "bmi" in selected:
    bmi_median = float(X_train["bmi"].median())
    X_train = X_train.fillna({"bmi": bmi_median})
    X_test = X_test.fillna({"bmi": bmi_median})
    st.caption(f"체질량지수의 빈 값은 훈련용 중앙값 {bmi_median:.1f}로 채웠어요.")

# ---------- 모델 학습 ('분류 모델' 페이지와 같은 방법) ----------
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
lr.fit(X_train, y_train)

tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
tree.fit(X_train, y_train)

base = DummyClassifier(strategy="most_frequent")
base.fit(X_train, y_train)

models = [(LR_NAME, lr), (TREE_NAME, tree), (BASE_NAME, base)]
preds = {
    name: pd.Series(model.predict(X_test), index=X_test.index) for name, model in models
}


# ---------- 채점 도구 ----------
def confusion(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fp, fn, tn


def pct(x):
    return f"{x * 100:.2f}%"


def scores(tp, fp, fn, tn):
    """네 칸에서 정확도·재현율·정밀도·F1을 구한다. 분모가 0이면 '계산할 수 없음'."""
    accuracy = pct((tp + tn) / (tp + fp + fn + tn))
    recall_val = tp / (tp + fn) if tp + fn > 0 else None  # 분모: 실제 뇌졸중인 사람 수
    precision_val = tp / (tp + fp) if tp + fp > 0 else None  # 분모: 뇌졸중이라 예측한 사람 수
    if precision_val is None or recall_val is None or precision_val + recall_val == 0:
        f1 = NA
    else:
        f1 = pct(2 * precision_val * recall_val / (precision_val + recall_val))
    return {
        "정확도": accuracy,
        "재현율": pct(recall_val) if recall_val is not None else NA,
        "정밀도": pct(precision_val) if precision_val is not None else NA,
        "F1": f1,
    }


def n(x):
    return f"{x:,}명"


cms = {name: confusion(y_test, preds[name]) for name, _ in models}

# ---------- 2. 혼동행렬 ----------
st.header("2. 혼동행렬")
st.write(
    f"테스트용 {len(y_test):,}명 가운데 뇌졸중인 사람은 {int(y_test.sum()):,}명이에요."
)
st.caption("행은 실제, 열은 예측이에요. TP·FP·FN·TN은 각각 칸의 이름이에요.")

cm_cols = st.columns(2)
for col, (name, _) in zip(cm_cols, models[:2]):
    tp, fp, fn, tn = cms[name]
    table = pd.DataFrame(
        [
            [f"TP · {n(tp)}", f"FN · {n(fn)}", n(tp + fn)],
            [f"FP · {n(fp)}", f"TN · {n(tn)}", n(fp + tn)],
            [n(tp + fp), n(fn + tn), n(tp + fp + fn + tn)],
        ],
        index=["실제 뇌졸중(양성)", "실제 아님(음성)", "합계"],
        columns=["예측 뇌졸중(양성)", "예측 아님(음성)", "합계"],
    )
    with col:
        st.subheader(name)
        st.table(table)

# ---------- 3. 네 가지 점수 ----------
st.header("3. 네 칸으로 구한 점수")
rows = []
for name, _ in models:
    tp, fp, fn, tn = cms[name]
    rows.append({"모델": name, **scores(tp, fp, fn, tn)})
st.dataframe(pd.DataFrame(rows), hide_index=True)
st.caption(
    "정확도 = (TP + TN) ÷ 전체 사람 수 · "
    "재현율 = TP ÷ (TP + FN), 분모는 실제 뇌졸중인 사람 수 · "
    "정밀도 = TP ÷ (TP + FP), 분모는 뇌졸중이라 예측한 사람 수 · "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율). "
    f"분모가 0이면 ‘{NA}’이라고 적어요."
)

# ---------- 4. 틀린 사람 목록 ----------
st.header("4. 모델이 틀린 사람들")
picked = st.radio("어느 모델의 목록을 볼까요?", [LR_NAME, TREE_NAME], horizontal=True)
p_tp, p_fp, p_fn, p_tn = cms[picked]
pred_picked = preds[picked]

test_rows = df[is_test]
missed = test_rows[(y_test == 1) & (pred_picked == 0)]  # 놓친 환자 (FN)
false_alarm = test_rows[(y_test == 0) & (pred_picked == 1)]  # 헛짚은 사람 (FP)


def list_table(rows_df):
    out = rows_df.sort_values(["age", "id"], ascending=[False, True])
    return pd.DataFrame(
        {
            "번호": out["id"],
            "나이": out["age"],
            "평균 혈당": out["avg_glucose_level"],
            "고혈압 여부": out["hypertension"].map({1: "있음", 0: "없음"}),
            "심장병 여부": out["heart_disease"].map({1: "있음", 0: "없음"}),
        }
    )


def show_list(title, rows_df):
    st.subheader(f"{title} · 전체 {len(rows_df):,}명")
    if len(rows_df) == 0:
        st.info("해당하는 사람이 없어요.")
        return
    shown = list_table(rows_df).head(MAX_ROWS)
    if len(rows_df) > MAX_ROWS:
        st.caption(f"나이가 많은 순서로 앞 {MAX_ROWS}명만 보여 줘요.")
    else:
        st.caption("나이가 많은 순서예요.")
    st.dataframe(
        shown,
        hide_index=True,
        column_config={
            "번호": st.column_config.NumberColumn("번호", format="%d"),
            "나이": st.column_config.NumberColumn("나이", format="%g"),
            "평균 혈당": st.column_config.NumberColumn("평균 혈당", format="%.2f"),
        },
    )


st.write(f"**{picked}**의 목록이에요.")
left, right = st.columns(2)
with left:
    show_list("놓친 환자 (실제 뇌졸중인데 아니라고 예측)", missed)
with right:
    show_list("헛짚은 사람 (아닌데 뇌졸중이라고 예측)", false_alarm)

same = len(missed) == p_fn and len(false_alarm) == p_fp
st.write(
    f"목록의 사람 수는 놓친 환자 {len(missed):,}명 = FN {p_fn:,}명, "
    f"헛짚은 사람 {len(false_alarm):,}명 = FP {p_fp:,}명으로 "
    + ("혼동행렬의 FN·FP 칸과 같아요." if same else "혼동행렬의 FN·FP 칸과 달라요. 확인이 필요해요.")
)
