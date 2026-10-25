import streamlit as st
import pandas as pd

st.set_page_config(page_title="운동 강도와 심박수 분석", page_icon="❤️", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(180deg, #f5f8ff 0%, #ffffff 55%);}
.title {font-size:2.25rem;font-weight:800;color:#183153;margin-bottom:.2rem;}
.subtitle {font-size:1rem;color:#64748b;margin-bottom:1rem;}
.question {padding:1rem 1.2rem;border-radius:14px;background:#edf5ff;
border-left:5px solid #3678c8;color:#183153;margin:.7rem 0 1.3rem 0;}
div[data-testid="stMetric"] {background:#fff;border:1px solid #e5eaf2;
padding:1rem;border-radius:14px;box-shadow:0 3px 12px rgba(24,49,83,.05);}
.note {color:#64748b;font-size:.92rem;margin-bottom:.8rem;}
</style>
""", unsafe_allow_html=True)

DATA_URL = "https://raw.githubusercontent.com/dltnqls7/dsproject__/refs/heads/main/subject101_preprocessed_1sec.csv"

# 활동명 표기 차이가 있어도 한국어로 변환하도록 정리
ACTIVITY_KOREAN = {
    "lying": "누워 있기", "lie": "누워 있기",
    "sitting": "앉아 있기", "sit": "앉아 있기",
    "standing": "서 있기", "stand": "서 있기",
    "walking": "걷기", "walk": "걷기",
    "running": "달리기", "run": "달리기",
    "cycling": "자전거 타기", "biking": "자전거 타기",
    "nordic walking": "노르딕 워킹", "nordic_walking": "노르딕 워킹",
    "ascending stairs": "계단 올라가기", "ascending_stairs": "계단 올라가기",
    "descending stairs": "계단 내려가기", "descending_stairs": "계단 내려가기",
    "vacuum cleaning": "청소기 돌리기", "vacuum_cleaning": "청소기 돌리기",
    "ironing": "다림질하기",
    "rope jumping": "줄넘기", "rope_jumping": "줄넘기",
    "house cleaning": "집안 청소", "watching tv": "TV 보기",
    "computer work": "컴퓨터 작업", "car driving": "자동차 운전"
}
INTENSITY_KOREAN = {
    "low": "낮음", "light": "낮음", "medium": "보통",
    "moderate": "보통", "high": "높음", "vigorous": "높음",
    "낮음": "낮음", "보통": "보통", "높음": "높음"
}

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

try:
    df = load_data()

    needed = ["time_seconds", "activity_name", "heart_rate_bpm", "MET", "intensity"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.error("CSV에 필요한 열이 없습니다: " + ", ".join(missing))
        st.stop()

    for col in ["time_seconds", "heart_rate_bpm", "MET"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time_seconds", "heart_rate_bpm", "activity_name"]).copy()

    def translate_activity(value):
        original = str(value).strip()
        key = " ".join(original.lower().replace("_", " ").replace("-", " ").split())
        return ACTIVITY_KOREAN.get(key, original)

    df["activity_name"] = df["activity_name"].apply(translate_activity)
    df["intensity"] = df["intensity"].astype(str).str.strip().str.lower().map(
        INTENSITY_KOREAN
    ).fillna(df["intensity"].astype(str))

    st.markdown('<div class="title">❤️ 운동 강도와 심박수 분석</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">센서 데이터로 살펴보는 신체 활동과 심박수의 관계</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="question"><b>🔎 핵심 질문</b><br>운동 강도가 높아질수록 심박수도 높아지는가?</div>',
        unsafe_allow_html=True
    )

    st.sidebar.title("📊 분석 설정")
    st.sidebar.caption("그래프에 표시할 활동을 선택하세요.")
    activities = sorted(df["activity_name"].dropna().unique().tolist())
    selected = st.sidebar.multiselect("활동 선택", activities, default=activities)
    filtered = df[df["activity_name"].isin(selected)].copy()
    if filtered.empty:
        st.warning("선택된 데이터가 없습니다. 왼쪽 메뉴에서 활동을 선택해 주세요.")
        st.stop()

    st.subheader("한눈에 보는 데이터")
    a, b, c, d = st.columns(4)
    a.metric("분석 데이터 수", f"{len(filtered):,}개")
    b.metric("평균 심박수", f"{filtered['heart_rate_bpm'].mean():.1f} bpm")
    c.metric("최대 심박수", f"{filtered['heart_rate_bpm'].max():.1f} bpm")
    d.metric("활동 종류", f"{filtered['activity_name'].nunique()}종")

    st.markdown("---")
    st.subheader("1. 시간에 따른 심박수 변화")
    st.markdown('<div class="note">시간이 흐르면서 심박수가 어떻게 변하는지 확인합니다.</div>', unsafe_allow_html=True)
