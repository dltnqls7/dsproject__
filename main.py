import streamlit as st
import pandas as pd

# --------------------------------------------------
# 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="운동 강도와 심박수 분석",
    page_icon="❤️",
    layout="wide"
)

# --------------------------------------------------
# 디자인
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #f5f8ff 0%, #ffffff 55%);
}
.title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #183153;
    margin-bottom: .2rem;
}
.subtitle {
    font-size: 1rem;
    color: #64748b;
    margin-bottom: 1rem;
}
.question {
    padding: 1rem 1.2rem;
    border-radius: 14px;
    background: #edf5ff;
    border-left: 5px solid #3678c8;
    color: #183153;
    margin: .7rem 0 1.3rem 0;
}
div[data-testid="stMetric"] {
    background: #fff;
    border: 1px solid #e5eaf2;
    padding: 1rem;
    border-radius: 14px;
    box-shadow: 0 3px 12px rgba(24,49,83,.05);
}
.note {
    color: #64748b;
    font-size: .92rem;
    margin-bottom: .8rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# GitHub CSV 주소
# --------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "dltnqls7/dsproject__/refs/heads/main/"
    "subject101_preprocessed.csv"
)

# --------------------------------------------------
# 활동명 한국어 번역
# --------------------------------------------------
ACTIVITY_KOREAN = {
    "lying": "누워 있기",
    "lying down": "누워 있기",
    "sitting": "앉아 있기",
    "sitting down": "앉아 있기",
    "standing": "서 있기",
    "standing still": "서 있기",
    "walking": "걷기",
    "running": "달리기",
    "cycling": "자전거 타기",
    "nordic walking": "노르딕 워킹",
    "ascending stairs": "계단 올라가기",
    "descending stairs": "계단 내려가기",
    "vacuum cleaning": "청소기 돌리기",
    "ironing": "다림질하기",
    "rope jumping": "줄넘기",
    "computer work": "컴퓨터 작업",
    "car driving": "자동차 운전",
    "house cleaning": "집안 청소",
    "watching tv": "TV 시청",
    "folding laundry": "빨래 개기",
    "playing soccer": "축구하기",
    "working": "작업하기",
    "ascending_stairs": "계단 올라가기",
    "descending_stairs": "계단 내려가기",
}

INTENSITY_KOREAN = {
    "low": "낮음",
    "light": "낮음",
    "medium": "보통",
    "moderate": "보통",
    "high": "높음",
    "vigorous": "높음",
    "낮음": "낮음",
    "보통": "보통",
    "높음": "높음",
}

# --------------------------------------------------
# GitHub에서 데이터 직접 읽기
# --------------------------------------------------
@st.cache_data
def load_data(url):
    return pd.read_csv(url)

def translate_activity(value):
    original = str(value).strip()

    key = (
        original.lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    key = " ".join(key.split())

    return ACTIVITY_KOREAN.get(key, original)

st.markdown(
    '<div class="title">❤️ 운동 강도와 심박수 분석</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    '센서 데이터로 살펴보는 신체 활동과 심박수의 관계'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="question">'
    '<b>🔎 핵심 질문</b><br>'
    '운동 강도가 높아질수록 심박수도 높아지는가?'
    '</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# 데이터 불러오기
# CSV 파일 업로드 기능 없이 GitHub에서 직접 읽음
# --------------------------------------------------
st.sidebar.title("📊 분석 설정")

try:
    df = load_data(DATA_URL)

except Exception as e:
    st.error("GitHub에서 데이터를 불러오지 못했습니다.")

    st.markdown(
        "GitHub의 CSV 주소가 정상적으로 열리는지 확인해 주세요."
    )

    st.code(DATA_URL)

    st.caption(f"오류 내용: {e}")

    st.stop()

# --------------------------------------------------
# 필요한 열 확인
# --------------------------------------------------
required_columns = [
    "time_seconds",
    "activity_name",
    "heart_rate_bpm",
    "MET",
    "intensity",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        "CSV에 필요한 열이 없습니다: "
        + ", ".join(missing_columns)
    )

    st.caption(
        "필요한 열: time_seconds, activity_name, "
        "heart_rate_bpm, MET, intensity"
    )

    st.stop()

# --------------------------------------------------
# 데이터 전처리
# --------------------------------------------------
for col in ["time_seconds", "heart_rate_bpm", "MET"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=[
        "time_seconds",
        "activity_name",
        "heart_rate_bpm",
    ]
).copy()

if df.empty:
    st.error("분석할 데이터가 없습니다.")
    st.stop()

# 활동명 한국어로 변경
df["activity_name"] = df["activity_name"].apply(
    translate_activity
)

# 운동 강도 한국어로 변경
df["intensity"] = (
    df["intensity"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(INTENSITY_KOREAN)
    .fillna(df["intensity"].astype(str))
)

st.success(
    f"GitHub에서 데이터를 불러왔습니다. "
    f"총 {len(df):,}개 행"
)

# --------------------------------------------------
# 사이드바 활동 필터
# --------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("활동 필터")

activities = sorted(
    df["activity_name"].dropna().unique().tolist()
)

selected = st.sidebar.multiselect(
    "분석할 활동을 선택하세요.",
    activities,
    default=activities
)

filtered = df[
    df["activity_name"].isin(selected)
].copy()

if filtered.empty:
    st.warning(
        "선택된 데이터가 없습니다. "
        "왼쪽 메뉴에서 활동을 선택해 주세요."
    )
    st.stop()

# --------------------------------------------------
# 요약 카드
# --------------------------------------------------
st.subheader("📌 한눈에 보는 데이터")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "분석 데이터 수",
    f"{len(filtered):,}개"
)

c2.metric(
    "평균 심박수",
    f"{filtered['heart_rate_bpm'].mean():.1f} bpm"
)

c3.metric(
    "최대 심박수",
    f"{filtered['heart_rate_bpm'].max():.1f} bpm"
)

c4.metric(
    "활동 종류",
    f"{filtered['activity_name'].nunique()}종"
)

# --------------------------------------------------
# 그래프 1: 시간에 따른 심박수
# --------------------------------------------------
st.markdown("---")
st.subheader("1. 시간에 따른 심박수 변화")

st.markdown(
    '<div class="note">'
    '시간이 흐르면서 심박수가 어떻게 변하는지 확인합니다.'
    '</div>',
    unsafe_allow_html=True
)

time_data = (
    filtered
    .sort_values("time_seconds")
    [["time_seconds", "heart_rate_bpm"]]
    .rename(
        columns={
            "time_seconds": "시간(초)",
            "heart_rate_bpm": "심박수(bpm)",
        }
    )
)

st.line_chart(
    time_data,
    x="시간(초)",
    y="심박수(bpm)",
    use_container_width=True
)

# --------------------------------------------------
# 그래프 2: 활동별 평균 심박수
# --------------------------------------------------
st.markdown("---")
st.subheader("2. 활동별 평균 심박수")

st.markdown(
    '<div class="note">'
    '활동 종류에 따른 평균 심박수를 비교합니다.'
    '</div>',
    unsafe_allow_html=True
)

activity_summary = (
    filtered
    .groupby("activity_name", as_index=False)
    ["heart_rate_bpm"]
    .mean()
    .rename(
        columns={
            "activity_name": "활동",
            "heart_rate_bpm": "평균 심박수(bpm)",
        }
    )
    .sort_values(
        "평균 심박수(bpm)",
        ascending=False
    )
    .set_index("활동")
)

st.bar_chart(
    activity_summary,
    y="평균 심박수(bpm)",
    use_container_width=True
)

# --------------------------------------------------
# 그래프 3: MET와 심박수 관계
# --------------------------------------------------
st.markdown("---")
st.subheader("3. 운동 강도(MET)와 심박수의 관계")

st.markdown(
    '<div class="note">'
    'MET는 신체 활동에 따른 에너지 소비 수준을 나타내는 지표입니다. '
    '오른쪽일수록 MET가 높고, 위쪽일수록 심박수가 높습니다.'
    '</div>',
    unsafe_allow_html=True
)

met_data = filtered.dropna(
    subset=["MET", "heart_rate_bpm"]
)

if not met_data.empty:
    met_summary = (
        met_data
        .groupby(
            ["activity_name", "MET"],
            as_index=False
        )["heart_rate_bpm"]
        .mean()
        .rename(
            columns={
                "activity_name": "활동",
                "MET": "운동 강도(MET)",
                "heart_rate_bpm": "평균 심박수(bpm)",
            }
        )
    )

    st.scatter_chart(
        met_summary,
        x="운동 강도(MET)",
        y="평균 심박수(bpm)",
        color="활동",
        use_container_width=True
    )

else:
    st.info(
        "MET 값이 있는 데이터가 없어 "
        "그래프를 표시할 수 없습니다."
    )

# --------------------------------------------------
# 그래프 4: 강도별 평균 심박수
# --------------------------------------------------
st.markdown("---")
st.subheader("4. 운동 강도 단계별 평균 심박수")

st.markdown(
    '<div class="note">'
    '기록된 운동 강도 단계별 평균 심박수를 비교합니다.'
    '</div>',
    unsafe_allow_html=True
)

intensity_summary = (
    filtered
    .groupby("intensity", as_index=False)
    ["heart_rate_bpm"]
    .mean()
    .rename(
        columns={
            "intensity": "운동 강도",
            "heart_rate_bpm": "평균 심박수(bpm)",
        }
    )
)

intensity_order = {
    "낮음": 0,
    "보통": 1,
    "높음": 2,
}

intensity_summary["_순서"] = (
    intensity_summary["운동 강도"]
    .map(intensity_order)
    .fillna(99)
)

intensity_summary = (
    intensity_summary
    .sort_values("_순서")
    .drop(columns="_순서")
)

st.bar_chart(
    intensity_summary.set_index("운동 강도"),
    y="평균 심박수(bpm)",
    use_container_width=True
)

# --------------------------------------------------
# 활동별 통계표
# --------------------------------------------------
st.markdown("---")

with st.expander("📋 활동별 통계표 보기"):
    table = (
        filtered
        .groupby("activity_name")
        .agg(
            데이터수=("heart_rate_bpm", "size"),
            평균심박수=("heart_rate_bpm", "mean"),
            최저심박수=("heart_rate_bpm", "min"),
            최고심박수=("heart_rate_bpm", "max"),
            평균MET=("MET", "mean"),
        )
        .reset_index()
        .rename(
            columns={
                "activity_name": "활동",
            }
        )
    )

    for col in [
        "평균심박수",
        "최저심박수",
        "최고심박수",
    ]:
        table[col] = table[col].round(1)

    table["평균MET"] = table["평균MET"].round(2)

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# 결과 해석 시 주의할 점
# --------------------------------------------------
st.markdown("---")
st.subheader("📌 결과 해석 시 주의할 점")

st.info(
    "이 데이터는 한 명의 참가자(Subject 101)의 기록이므로 "
    "모든 사람에게 일반화하기 어렵습니다. "
    "심박수는 운동 강도 외에도 개인의 체력, 휴식 상태, "
    "센서 오차 등의 영향을 받을 수 있습니다. "
    "따라서 그래프에서 관계가 나타나더라도 "
    "운동 강도만이 심박수 변화의 원인이라고 "
    "단정할 수 없습니다."
)

st.caption(
    "데이터 출처: PAMAP2 Physical Activity Monitoring Dataset "
    "· 활동명은 화면 표시용으로 한국어 번역"
)
