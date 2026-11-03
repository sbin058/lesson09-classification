import math
import random

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_URL = "https://raw.githubusercontent.com/sbin058/data/refs/heads/main/stroke.csv"

st.set_page_config(page_title="성능 높이기 - 뇌졸중 예측 실습실", page_icon="⚖️", layout="wide")

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

NA = "계산할 수 없음"
BLUE = "#1f77b4"  # 실제 아님
RED = "#d62728"  # 실제 뇌졸중(양성)
BEFORE_COLOR = "#ff7f0e"  # 가중치를 맞추기 전 경계
AFTER_COLOR = "#2ca02c"  # 가중치를 맞춘 뒤 경계


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("⚖️ 성능 높이기")
st.caption(
    "뇌졸중 예측 실습실 · 뇌졸중을 겪은 사람이 아주 드물 때, 양쪽의 가중치를 맞추면 결과가 어떻게 달라지는지 봐요."
)

try:
    raw = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

# ---------- 1. 입력 속성 고르기 ('분류 모델' 페이지와 같은 방법) ----------
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

# ---------- 데이터 나누기 ('채점표' 페이지와 같은 방법) ----------
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

n_pos_train = int(y_train.sum())
n_neg_train = len(y_train) - n_pos_train
weight_ratio = n_neg_train / n_pos_train if n_pos_train else float("nan")

st.write(
    f"훈련용 {len(X_train):,}명 가운데 뇌졸중인 사람은 {n_pos_train:,}명뿐이에요. "
    f"가중치를 맞추면 훈련할 때 뇌졸중인 사람 1명이 아닌 사람 1명의 약 {weight_ratio:.1f}배로 계산돼요. "
    "테스트용 데이터는 그대로이고, 훈련 방식만 달라져요."
)


# ---------- 2. 모델 학습: 가중치를 맞추기 전과 맞춘 뒤 ----------
def make_models(class_weight):
    lr_model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight=class_weight)
    )
    tree_model = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=5, random_state=42, class_weight=class_weight
    )
    lr_model.fit(X_train, y_train)
    tree_model.fit(X_train, y_train)
    return lr_model, tree_model


lr_before, tree_before = make_models(None)
lr_after, tree_after = make_models("balanced")


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
    return [
        accuracy,
        pct(recall_val) if recall_val is not None else NA,
        pct(precision_val) if precision_val is not None else NA,
        f1,
    ]


def people(x):
    return f"{x:,}명"


def predict_cm(model):
    pred = pd.Series(model.predict(X_test), index=X_test.index)
    return confusion(y_test, pred)


st.header("2. 가중치를 맞추기 전과 맞춘 뒤")
st.write(
    f"테스트용 {len(y_test):,}명 가운데 뇌졸중인 사람은 {int(y_test.sum()):,}명이에요. "
    "뇌졸중을 양성으로 두고, 표는 모두 테스트 데이터로 채점한 결과예요."
)

compare_cols = st.columns(2)
for col, (name, m_before, m_after) in zip(
    compare_cols,
    [(LR_NAME, lr_before, lr_after), (TREE_NAME, tree_before, tree_after)],
):
    b_tp, b_fp, b_fn, b_tn = predict_cm(m_before)
    a_tp, a_fp, a_fn, a_tn = predict_cm(m_after)
    table = pd.DataFrame(
        {
            "가중치 맞추기 전": [people(b_tp), people(b_fp), people(b_fn), people(b_tn)]
            + scores(b_tp, b_fp, b_fn, b_tn),
            "가중치 맞춘 뒤": [people(a_tp), people(a_fp), people(a_fn), people(a_tn)]
            + scores(a_tp, a_fp, a_fn, a_tn),
        },
        index=[
            "TP (찾아낸 환자)",
            "FP (헛짚은 사람)",
            "FN (놓친 환자)",
            "TN (아님을 맞힌 사람)",
            "정확도",
            "재현율",
            "정밀도",
            "F1",
        ],
    )
    with col:
        st.subheader(name)
        st.table(table)
        st.write(
            f"찾아낸 환자 {b_tp:,}명 → {a_tp:,}명 · "
            f"놓친 환자 {b_fn:,}명 → {a_fn:,}명 · "
            f"헛짚은 사람 {b_fp:,}명 → {a_fp:,}명"
        )
st.caption(
    "재현율 = TP ÷ (TP + FN), 정밀도 = TP ÷ (TP + FP), "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율). "
    f"분모가 0이면 ‘{NA}’이라고 적어요."
)


# ---------- 경계선·경계 평면 계산 도구 ----------
def view_range(col):
    if col in BINARY:
        return -0.5, 1.5
    lo, hi = float(X_test[col].min()), float(X_test[col].max())
    pad = (hi - lo) * 0.05 or 1.0
    return lo - pad, hi + pad


def fmt_fixed(col, value):
    if col in BINARY:
        return f"{KOREAN[col]} {value:g}({'있음' if value >= 0.5 else '없음'})"
    return f"{KOREAN[col]} {value:.2f}"


def linear_form(model, axis_cols, fixed):
    """축이 아닌 속성을 fixed 값에 세워 두면 결정함수는 축 속성에 대한 직선(평면)이 된다.
    결정함수 = 계수·축값 + 상수 = 0 인 곳이 확률 0.5인 곳이다."""
    base = {c: 0.0 for c in axis_cols}
    rows = [base] + [{**base, c: 1.0} for c in axis_cols]
    frame = pd.DataFrame([{c: r.get(c, fixed[c]) for c in selected} for r in rows])
    values = model.decision_function(frame).tolist()
    const = values[0]
    return [v - const for v in values[1:]], const


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


def dot(u, v):
    return sum(p * q for p, q in zip(u, v))


def cross(u, v):
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def unit(u):
    m = math.sqrt(dot(u, u))
    return tuple(p / m for p in u)


def plane_polygon(coefs, const, ranges):
    """a*x + b*y + c*z + const = 0 인 평면이 상자(ranges) 안을 지나는 다각형의 꼭짓점을 돌려준다."""
    a, b, c = coefs
    n = (a, b, c)
    if dot(n, n) < 1e-24:
        return None

    def f(p):
        return a * p[0] + b * p[1] + c * p[2] + const

    corners = [(x, y, z) for x in ranges[0] for y in ranges[1] for z in ranges[2]]
    pts = [p for p in corners if f(p) == 0]
    for p in corners:
        for q in corners:
            if p < q and sum(1 for i in range(3) if p[i] != q[i]) == 1:  # 상자의 모서리
                fp, fq = f(p), f(q)
                if fp * fq < 0:
                    t = fp / (fp - fq)
                    pts.append(tuple(p[i] + t * (q[i] - p[i]) for i in range(3)))
    uniq = []
    for p in pts:
        if all(max(abs(p[i] - q[i]) for i in range(3)) > 1e-9 for q in uniq):
            uniq.append(p)
    if len(uniq) < 3:
        return None
    center = tuple(sum(p[i] for p in uniq) / len(uniq) for i in range(3))
    axis = min(((1, 0, 0), (0, 1, 0), (0, 0, 1)), key=lambda e: abs(dot(n, e)))
    u = unit(cross(n, axis))
    v = unit(cross(n, u))

    def angle(p):
        d = tuple(p[i] - center[i] for i in range(3))
        return math.atan2(dot(d, v), dot(d, u))

    uniq.sort(key=angle)
    return uniq


def jitter_list(col, count, rng):
    """0/1 속성은 점이 겹치지 않도록 그림에서만 살짝 흩뿌린다."""
    return [rng.uniform(-0.08, 0.08) if col in BINARY else 0.0 for _ in range(count)]


# ---------- 3. 2차원 산점도 ----------
st.header("3. 테스트 데이터 산점도와 0.5 경계선")
st.caption(
    f"{LR_NAME}이 뇌졸중일 확률 0.5로 가르는 선을 가중치를 맞추기 전과 맞춘 뒤로 겹쳐 그려요."
)
ax1, ax2 = st.columns(2)
with ax1:
    x_col = st.selectbox("가로축", selected, index=0, format_func=lambda c: KOREAN[c])
with ax2:
    y_col = st.selectbox("세로축", selected, index=1, format_func=lambda c: KOREAN[c])

if x_col == y_col:
    st.info("가로축과 세로축은 서로 다른 속성으로 골라 주세요.")
else:
    xn, yn = KOREAN[x_col], KOREAN[y_col]
    xr, yr = view_range(x_col), view_range(y_col)
    fixed2 = {c: float(X_test[c].median()) for c in selected}
    others2 = [c for c in selected if c not in (x_col, y_col)]

    fig = go.Figure()
    rng = random.Random(0)
    jit_x = jitter_list(x_col, len(X_test), rng)
    jit_y = jitter_list(y_col, len(X_test), rng)
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
                mode="markers",
                name=label,
                marker=dict(color=color, size=size, opacity=0.7, line=dict(width=0.5, color="white")),
                customdata=sub[[x_col, y_col]].values.tolist(),
                hovertemplate=(
                    f"{xn}: %{{customdata[0]:.4g}}<br>{yn}: %{{customdata[1]:.4g}}"
                    f"<extra>{label}</extra>"
                ),
            )
        )

    line_notes = []
    for label, model, color, dash in [
        ("가중치 맞추기 전", lr_before, BEFORE_COLOR, "dash"),
        ("가중치 맞춘 뒤", lr_after, AFTER_COLOR, "solid"),
    ]:
        (a, b), c0 = linear_form(model, [x_col, y_col], fixed2)
        segment = clip_line(a, b, c0, xr, yr)
        color_name = "주황색 점선" if color == BEFORE_COLOR else "초록색 실선"
        if segment is not None:
            (lx0, ly0), (lx1, ly1) = segment
            fig.add_trace(
                go.Scatter(
                    x=[lx0, lx1],
                    y=[ly0, ly1],
                    mode="lines",
                    name=f"{label} 0.5 경계선",
                    line=dict(color=color, width=3, dash=dash),
                    hoverinfo="skip",
                )
            )
            if abs(b) * (yr[1] - yr[0]) >= abs(a) * (xr[1] - xr[0]):
                side = "위쪽" if b > 0 else "아래쪽"
            else:
                side = "오른쪽" if a > 0 else "왼쪽"
            line_notes.append(
                f"{color_name}({label})은 그림 안에 있어요. 선의 {side}에서는 뇌졸중이라고, 반대쪽에서는 아님이라고 답해요."
            )
        else:
            center_f = c0 + a * (sum(xr) / 2) + b * (sum(yr) / 2)
            where = (
                "뇌졸중일 확률이 어디서나 0.5보다 낮아서 모두 아님이라고 답해요."
                if center_f < 0
                else "뇌졸중일 확률이 어디서나 0.5보다 높아서 모두 뇌졸중이라고 답해요."
            )
            line_notes.append(
                f"{color_name}({label})은 이 그림 범위 밖에 있어서 그리지 못했어요. 그림 안에서는 {where}"
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

    if others2:
        st.write(
            "두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: "
            + ", ".join(fmt_fixed(c, fixed2[c]) for c in others2)
        )
    else:
        st.write("고른 속성이 두 개뿐이라 따로 세워 둔 속성은 없어요.")
    for note in line_notes:
        st.write(note)
    if x_col in BINARY or y_col in BINARY:
        st.caption("0/1 값인 속성은 점이 겹치지 않도록 살짝 흩뿌려 찍었어요.")

# ---------- 4. 3차원 산점도 ----------
st.header("4. 3차원 산점도와 0.5 평면")
if len(selected) < 3:
    st.info("입력 속성을 세 개 이상 고르면 3차원 산점도도 볼 수 있어요.")
else:
    st.caption(
        f"{LR_NAME}이 뇌졸중일 확률 0.5로 가르는 평면을 가중치를 맞추기 전과 맞춘 뒤로 겹쳐 그려요. "
        "마우스로 끌어서 돌려 볼 수 있어요."
    )
    d1, d2, d3 = st.columns(3)
    with d1:
        x3 = st.selectbox("3차원 가로축", selected, index=0, format_func=lambda c: KOREAN[c])
    with d2:
        y3 = st.selectbox("3차원 세로축", selected, index=1, format_func=lambda c: KOREAN[c])
    with d3:
        z3 = st.selectbox("3차원 높이축", selected, index=2, format_func=lambda c: KOREAN[c])

    if len({x3, y3, z3}) < 3:
        st.info("세 축은 서로 다른 속성으로 골라 주세요.")
    else:
        axes3 = [x3, y3, z3]
        names3 = [KOREAN[c] for c in axes3]
        ranges3 = [view_range(c) for c in axes3]
        fixed3 = {c: float(X_test[c].median()) for c in selected}
        others3 = [c for c in selected if c not in axes3]

        fig3 = go.Figure()
        rng3 = random.Random(1)
        pts3 = X_test[axes3].copy()
        for i, c in enumerate(axes3):
            pts3[f"j{i}"] = jitter_list(c, len(X_test), rng3)
        pts3["y"] = y_test.tolist()
        for cls, color, label, size in [
            (0, BLUE, "실제 아님", 3),
            (1, RED, "실제 뇌졸중", 5),
        ]:
            sub = pts3[pts3["y"] == cls]
            fig3.add_trace(
                go.Scatter3d(
                    x=(sub[x3] + sub["j0"]).tolist(),
                    y=(sub[y3] + sub["j1"]).tolist(),
                    z=(sub[z3] + sub["j2"]).tolist(),
                    mode="markers",
                    name=label,
                    marker=dict(color=color, size=size, opacity=0.75),
                    customdata=sub[axes3].values.tolist(),
                    hovertemplate=(
                        f"{names3[0]}: %{{customdata[0]:.4g}}<br>"
                        f"{names3[1]}: %{{customdata[1]:.4g}}<br>"
                        f"{names3[2]}: %{{customdata[2]:.4g}}<extra>{label}</extra>"
                    ),
                )
            )

        plane_notes = []
        for label, model, color in [
            ("가중치 맞추기 전", lr_before, BEFORE_COLOR),
            ("가중치 맞춘 뒤", lr_after, AFTER_COLOR),
        ]:
            coefs, c0 = linear_form(model, axes3, fixed3)
            polygon = plane_polygon(coefs, c0, ranges3)
            color_name = "주황색 평면" if color == BEFORE_COLOR else "초록색 평면"
            if polygon is not None:
                m = len(polygon)
                fig3.add_trace(
                    go.Mesh3d(
                        x=[p[0] for p in polygon],
                        y=[p[1] for p in polygon],
                        z=[p[2] for p in polygon],
                        i=[0] * (m - 2),
                        j=list(range(1, m - 1)),
                        k=list(range(2, m)),
                        color=color,
                        opacity=0.35,
                        name=f"{label} 0.5 평면",
                        showlegend=True,
                        hoverinfo="skip",
                    )
                )
                plane_notes.append(f"{color_name}({label})은 그림 상자 안을 지나요.")
            else:
                center_f = c0 + sum(
                    coefs[i] * (sum(ranges3[i]) / 2) for i in range(3)
                )
                where = (
                    "뇌졸중일 확률이 어디서나 0.5보다 낮아서 모두 아님이라고 답해요."
                    if center_f < 0
                    else "뇌졸중일 확률이 어디서나 0.5보다 높아서 모두 뇌졸중이라고 답해요."
                )
                plane_notes.append(
                    f"{color_name}({label})은 이 그림 상자 밖에 있어서 그리지 못했어요. 상자 안에서는 {where}"
                )

        def axis_dict(col, rng_):
            d = dict(title=KOREAN[col], range=list(rng_))
            if col in BINARY:
                d.update(tickvals=[0, 1], ticktext=["없음", "있음"])
            return d

        fig3.update_layout(
            title="테스트 데이터: 실제 뇌졸중 여부",
            height=680,
            legend=dict(orientation="h", yanchor="top", y=0.0, x=0),
            margin=dict(l=0, r=0, t=50, b=0),
            scene=dict(
                xaxis=axis_dict(x3, ranges3[0]),
                yaxis=axis_dict(y3, ranges3[1]),
                zaxis=axis_dict(z3, ranges3[2]),
                aspectmode="cube",
            ),
        )
        st.plotly_chart(fig3)

        if others3:
            st.write(
                "세 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: "
                + ", ".join(fmt_fixed(c, fixed3[c]) for c in others3)
            )
        else:
            st.write("고른 속성이 세 개뿐이라 따로 세워 둔 속성은 없어요.")
        for note in plane_notes:
            st.write(note)
        if any(c in BINARY for c in axes3):
            st.caption("0/1 값인 속성은 점이 겹치지 않도록 살짝 흩뿌려 찍었어요.")
