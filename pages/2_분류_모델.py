import random

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_URL = "https://raw.githubusercontent.com/sbin058/data/refs/heads/main/stroke.csv"

st.set_page_config(page_title="분류 모델 - 뇌졸중 예측 실습실", page_icon="🤖", layout="wide")

KOREAN = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
ALL_FEATURES = list(KOREAN)
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]
BINARY = {"hypertension", "heart_disease"}  # 0/1 값을 가진 속성

LR_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "기준 모델(입력을 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)"

BLUE = "#1f77b4"  # 아님
RED = "#d62728"  # 뇌졸중(양성)

st.markdown(
    """
    <style>
    .acc-card { border: 1px solid rgba(128,128,128,0.45); border-radius: 12px; padding: 1rem 1.1rem; }
    .acc-name { font-size: 0.95rem; font-weight: 600; min-height: 2.8rem; }
    .acc-big { font-size: 3rem; font-weight: 800; line-height: 1.1; }
    .acc-big-label { font-size: 0.85rem; opacity: 0.7; margin-bottom: 0.7rem; }
    .acc-small { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 0.2rem 0.8rem;
                 font-size: 0.8rem; opacity: 0.85; border-top: 1px solid rgba(128,128,128,0.35);
                 padding-top: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def card_html(name, train_acc, test_acc):
    return (
        '<div class="acc-card">'
        f'<div class="acc-name">{name}</div>'
        f'<div class="acc-big">{test_acc * 100:.2f}%</div>'
        '<div class="acc-big-label">테스트 데이터 정확도</div>'
        '<div class="acc-small">'
        f"<span>훈련 데이터 {train_acc * 100:.2f}%</span>"
        f"<span>테스트 데이터 {test_acc * 100:.2f}%</span>"
        "</div></div>"
    )


st.title("🤖 분류 모델")
st.caption("뇌졸중 예측 실습실 · stroke가 1이면 뇌졸중, 0이면 아님이에요. 뇌졸중을 양성으로 둬요.")

try:
    raw = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

# ---------- 1. 입력 속성 고르기 ----------
st.header("1. 입력으로 사용할 속성 고르기")
chosen = st.multiselect(
    "입력 속성 (두 개 이상)",
    options=ALL_FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda c: KOREAN[c],
)
selected = [c for c in ALL_FEATURES if c in chosen]  # 항상 같은 순서로
if len(selected) < 2:
    st.info("입력 속성을 두 개 이상 골라 주세요. 두 개 이상 고르면 모델을 만들어 보여 드려요.")
    st.stop()

# ---------- 2. 데이터 나누기 ----------
df = raw.sort_values("id", kind="mergesort").reset_index(drop=True)
is_test = (df.index % 10) < 3  # 열 명씩 묶어 앞 세 명은 테스트용
X_all = df[selected].copy()
y_all = df["stroke"]
X_train, X_test = X_all[~is_test], X_all[is_test]
y_train, y_test = y_all[~is_test], y_all[is_test]

st.header("2. 데이터 나누기")
st.write(
    f"번호(id) 순으로 정렬해 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 고정했어요. "
    f"**훈련용 {len(X_train):,}명** (뇌졸중 {int(y_train.sum()):,}명) · "
    f"**테스트용 {len(X_test):,}명** (뇌졸중 {int(y_test.sum()):,}명)"
)

if "bmi" in selected:
    bmi_median = float(X_train["bmi"].median())
    n_tr_missing = int(X_train["bmi"].isna().sum())
    n_te_missing = int(X_test["bmi"].isna().sum())
    X_train = X_train.fillna({"bmi": bmi_median})
    X_test = X_test.fillna({"bmi": bmi_median})
    st.caption(
        f"체질량지수의 빈 값(훈련용 {n_tr_missing:,}명, 테스트용 {n_te_missing:,}명)은 "
        f"훈련용 중앙값 {bmi_median:.1f}로 채웠어요."
    )

# ---------- 3. 모델 만들기 ----------
# 로지스틱 회귀: 훈련용으로만 크기 맞추기(표준화)와 학습을 해요.
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
lr.fit(X_train, y_train)

# 의사결정트리: 질문은 세 번까지, 마지막 마디에는 최소 5명, 난수 고정.
# (트리는 값의 크기를 맞춰도 질문이 같으므로, 질문을 원래 단위로 읽을 수 있게 원래 값을 써요.)
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
tree.fit(X_train, y_train)

# 기준 모델: 입력을 보지 않고 훈련용에서 많은 쪽으로만 답해요.
base = DummyClassifier(strategy="most_frequent")
base.fit(X_train, y_train)

st.header("3. 모델 정확도 비교")
cards = [
    (LR_NAME, lr),
    (TREE_NAME, tree),
    (BASE_NAME, base),
]
cols = st.columns(3)
for col, (name, model) in zip(cols, cards):
    with col:
        st.markdown(
            card_html(name, model.score(X_train, y_train), model.score(X_test, y_test)),
            unsafe_allow_html=True,
        )
st.caption(
    "의사결정트리는 질문을 최대 3번까지만 던지고, 마지막 마디에 사람이 5명 미만으로 남게 나누지 않아요. "
    "난수는 42로 고정했어요."
)

# ---------- 4. 산점도 ----------
st.header("4. 테스트 데이터 산점도와 모델이 가르는 선")
ax1, ax2 = st.columns(2)
with ax1:
    x_col = st.selectbox("가로축", selected, index=0, format_func=lambda c: KOREAN[c])
with ax2:
    y_col = st.selectbox("세로축", selected, index=1, format_func=lambda c: KOREAN[c])


def view_range(col):
    if col in BINARY:
        return -0.5, 1.5
    lo, hi = float(X_test[col].min()), float(X_test[col].max())
    pad = (hi - lo) * 0.05 or 1.0
    return lo - pad, hi + pad


def clip_line(a, b, c, xr, yr):
    """a*x + b*y + c = 0 인 선이 그림 범위 안을 지나는 두 끝점을 돌려준다. 안 지나면 None."""
    x0, x1 = xr
    y0, y1 = yr
    tol = 1e-9 * max(x1 - x0, y1 - y0)
    pts = []
    if abs(b) > 1e-12:
        for xv in (x0, x1):
            yv = -(c + a * xv) / b
            if y0 - tol <= yv <= y1 + tol:
                pts.append((xv, yv))
    if abs(a) > 1e-12:
        for yv in (y0, y1):
            xv = -(c + b * yv) / a
            if x0 - tol <= xv <= x1 + tol:
                pts.append((xv, yv))
    uniq = []
    for p in pts:
        if all(abs(p[0] - q[0]) > tol or abs(p[1] - q[1]) > tol for q in uniq):
            uniq.append(p)
    if len(uniq) < 2:
        return None
    uniq.sort()
    return uniq[0], uniq[-1]


def fmt_fixed(col, value):
    if col in BINARY:
        return f"{KOREAN[col]} {value:g}({'있음' if value >= 0.5 else '없음'})"
    return f"{KOREAN[col]} {value:.2f}"


if x_col == y_col:
    st.info("가로축과 세로축은 서로 다른 속성으로 골라 주세요.")
else:
    xn, yn = KOREAN[x_col], KOREAN[y_col]
    xr, yr = view_range(x_col), view_range(y_col)
    fixed = {c: float(X_test[c].median()) for c in selected}
    others = [c for c in selected if c not in (x_col, y_col)]

    def make_frame(xs_, ys_):
        data = {}
        for c in selected:
            if c == x_col:
                data[c] = xs_
            elif c == y_col:
                data[c] = ys_
            else:
                data[c] = [fixed[c]] * len(xs_)
        return pd.DataFrame(data)

    # (1) 로지스틱 회귀의 0.5 경계선: 결정함수 = 0 인 선. 결정함수는 입력에 대해 직선이므로
    #     세 점에서 계수를 읽어 낸다.
    f_vals = lr.decision_function(make_frame([0.0, 1.0, 0.0], [0.0, 0.0, 1.0])).tolist()
    c0 = f_vals[0]
    a = f_vals[1] - c0
    b = f_vals[2] - c0
    segment = clip_line(a, b, c0, xr, yr)

    # (2) 의사결정트리가 나눈 칸: 그림 범위를 촘촘한 격자로 나누어 트리의 답을 칠한다.
    N = 120
    gx = [xr[0] + (xr[1] - xr[0]) * i / (N - 1) for i in range(N)]
    gy = [yr[0] + (yr[1] - yr[0]) * i / (N - 1) for i in range(N)]
    grid_x = gx * N
    grid_y = [v for v in gy for _ in range(N)]
    pred = tree.predict(make_frame(grid_x, grid_y)).tolist()
    z = [pred[i * N:(i + 1) * N] for i in range(N)]
    present = set(pred)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=gx, y=gy, z=z, zmin=0, zmax=1, showscale=False, hoverinfo="skip",
            colorscale=[[0, "rgba(31,119,180,0.13)"], [1, "rgba(214,39,40,0.16)"]],
        )
    )
    # 칠한 칸의 범례(점 없는 표식)
    for cls, color, label in [
        (0, "rgba(31,119,180,0.35)", "의사결정트리 칸: 아님이라고 답함"),
        (1, "rgba(214,39,40,0.35)", "의사결정트리 칸: 뇌졸중이라고 답함"),
    ]:
        if cls in present:
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None], mode="markers", name=label,
                    marker=dict(symbol="square", size=14, color=color),
                )
            )

    # 테스트 데이터 점 (0/1 속성은 겹치지 않게 살짝 흩뿌려 그림)
    rng = random.Random(0)
    jit_x = [rng.uniform(-0.08, 0.08) if x_col in BINARY else 0.0 for _ in range(len(X_test))]
    jit_y = [rng.uniform(-0.08, 0.08) if y_col in BINARY else 0.0 for _ in range(len(X_test))]
    pts = X_test[[x_col, y_col]].copy()
    pts["jx"] = jit_x
    pts["jy"] = jit_y
    pts["y"] = y_test.tolist()
    for cls, color, label, size in [
        (0, BLUE, "실제 아님", 6),
        (1, RED, "실제 뇌졸중", 8),
    ]:
        sub = pts[pts["y"] == cls]
        fig.add_trace(
            go.Scatter(
                x=(sub[x_col] + sub["jx"]).tolist(),
                y=(sub[y_col] + sub["jy"]).tolist(),
                mode="markers", name=label,
                marker=dict(color=color, size=size, opacity=0.7, line=dict(width=0.5, color="white")),
                customdata=sub[[x_col, y_col]].values.tolist(),
                hovertemplate=(
                    f"{xn}: %{{customdata[0]:.4g}}<br>{yn}: %{{customdata[1]:.4g}}"
                    f"<extra>{label}</extra>"
                ),
            )
        )

    if segment is not None:
        (lx0, ly0), (lx1, ly1) = segment
        fig.add_trace(
            go.Scatter(
                x=[lx0, lx1], y=[ly0, ly1], mode="lines",
                name="로지스틱 회귀 0.5 경계선",
                line=dict(color="black", width=3, dash="dash"), hoverinfo="skip",
            )
        )

    fig.update_layout(
        title="테스트 데이터: 실제 뇌졸중 여부",
        height=580,
        legend=dict(orientation="h", yanchor="top", y=-0.15, x=0),
        margin=dict(t=60, b=40),
    )
    fig.update_xaxes(title=xn, range=list(xr))
    fig.update_yaxes(title=yn, range=list(yr))
    if x_col in BINARY:
        fig.update_xaxes(tickvals=[0, 1], ticktext=["없음", "있음"])
    if y_col in BINARY:
        fig.update_yaxes(tickvals=[0, 1], ticktext=["없음", "있음"])
    st.plotly_chart(fig)

    # 그림 아래 설명 글
    if others:
        st.write(
            "두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: "
            + ", ".join(fmt_fixed(c, fixed[c]) for c in others)
        )
    else:
        st.write("고른 속성이 두 개뿐이라 따로 세워 둔 속성은 없어요.")

    if segment is not None:
        if abs(b) * (yr[1] - yr[0]) >= abs(a) * (xr[1] - xr[0]):
            side = "위쪽" if b > 0 else "아래쪽"
        else:
            side = "오른쪽" if a > 0 else "왼쪽"
        st.write(
            f"점선은 {LR_NAME}이 뇌졸중일 확률 0.5로 가르는 선이에요. "
            f"선의 {side}에서는 뇌졸중이라고, 반대쪽에서는 아님이라고 답해요."
        )
    else:
        center_f = c0 + a * (sum(xr) / 2) + b * (sum(yr) / 2)
        if center_f < 0:
            where = "뇌졸중일 확률이 어디서나 0.5보다 낮아서 모두 아님이라고 답해요."
        else:
            where = "뇌졸중일 확률이 어디서나 0.5보다 높아서 모두 뇌졸중이라고 답해요."
        st.write(
            f"{LR_NAME}이 확률 0.5로 가르는 선은 이 그림 범위 밖에 있어서 그리지 못했어요. "
            f"그림 안에서는 {where}"
        )
    st.write(
        "옅은 색으로 칠한 칸은 "
        f"{TREE_NAME}이 그림을 나눈 모습이에요."
        + (" 0/1 값인 속성은 점이 겹치지 않도록 살짝 흩뿌려 찍었어요." if (x_col in BINARY or y_col in BINARY) else "")
    )

# ---------- 5. 의사결정트리가 던진 질문 ----------
st.header(f"5. {TREE_NAME}이 던진 질문")

t = tree.tree_
classes = tree.classes_.tolist()
leaf_ids = pd.DataFrame({"leaf": tree.apply(X_train), "y": y_train.tolist()})
pos_in_leaf = leaf_ids.groupby("leaf")["y"].sum().to_dict()
pos_cache = {}


def positives(node):
    """그 마디에 온 훈련용 사람 가운데 실제 뇌졸중인 사람 수."""
    if node not in pos_cache:
        left, right = int(t.children_left[node]), int(t.children_right[node])
        if left == -1:
            pos_cache[node] = int(pos_in_leaf.get(node, 0))
        else:
            pos_cache[node] = positives(left) + positives(right)
    return pos_cache[node]


def stats_text(node):
    n = int(t.n_node_samples[node])
    p = positives(node)
    return f"훈련용 {n:,}명\\n뇌졸중 {p:,}명 ({p / n * 100:.1f}%)"


dot = [
    "digraph tree {",
    'graph [bgcolor="white", ranksep=0.55, nodesep=0.3];',
    'node [shape=box, style="rounded,filled", fontcolor="black", fontsize=12, margin="0.2,0.1"];',
    'edge [fontcolor="black", color="#555555", fontsize=12];',
]
leaf_answers = []
asked_idx = set()
for node in range(t.node_count):
    left, right = int(t.children_left[node]), int(t.children_right[node])
    if left == -1:  # 더 묻지 않고 답을 내는 마디
        values = t.value[node][0].tolist()
        answer = classes[values.index(max(values))]
        leaf_answers.append(answer)
        name = "뇌졸중" if answer == 1 else "아님"
        fill, border = ("#f6c9c9", RED) if answer == 1 else ("#c9dff1", BLUE)
        dot.append(
            f'n{node} [label="답: {name}\\n{stats_text(node)}", '
            f'fillcolor="{fill}", color="{border}", penwidth=2];'
        )
    else:
        feat_idx = int(t.feature[node])
        asked_idx.add(feat_idx)
        col = selected[feat_idx]
        if col in BINARY:
            question = f"{KOREAN[col]}이 있나요?"
            yes_child, no_child = right, left  # 0.5보다 크면 '있음'
        else:
            question = f"{KOREAN[col]} ≤ {float(t.threshold[node]):.1f} ?"
            yes_child, no_child = left, right
        dot.append(
            f'n{node} [label="{question}\\n{stats_text(node)}", '
            f'fillcolor="#eeeeee", color="#888888"];'
        )
        dot.append(f'n{node} -> n{yes_child} [label=" 예"];')
        dot.append(f'n{node} -> n{no_child} [label=" 아니요"];')
dot.append("}")

st.graphviz_chart("\n".join(dot))
st.caption("마디의 숫자는 훈련용 데이터 기준이에요. 색이 있는 마디는 더 묻지 않고 답을 내는 마디예요.")

n_leaves = len(leaf_answers)
n_neg = sum(1 for ans in leaf_answers if ans == 0)
st.write(f"답을 내는 마디는 모두 {n_leaves}칸이고, 그중 {n_neg}칸이 ‘아님’이라고 답해요.")

asked = [KOREAN[selected[i]] for i in range(len(selected)) if i in asked_idx]
not_asked = [KOREAN[selected[i]] for i in range(len(selected)) if i not in asked_idx]
if asked:
    tail = f" (묻지 않은 속성: {', '.join(not_asked)})" if not_asked else ""
    st.write(
        f"고른 속성 {len(selected)}개 가운데 이 나무가 실제로 물은 속성: {', '.join(asked)}{tail}"
    )
else:
    st.write(
        f"고른 속성 {len(selected)}개 가운데 이 나무가 실제로 물은 속성: 없음 "
        "(질문 없이 하나의 답만 내요)"
    )
