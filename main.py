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

st.markdown("""
<style>
.stApp {background: linear-gradient(180deg, #f5f8ff 0%, #ffffff 55%);}
.title {font-size:2.2rem;font-weight:800;color:#183153;margin-bottom:.2rem;}
.subtitle {font-size:1rem;color:#64748b;margin-bottom:1rem;}
.question {padding:1rem 1.2rem;border-radius:14px;background:#edf5ff;
border-left:5px solid #3678c8;color:#183153;margin:.7rem 0 1.3rem 0;}
div[data-testid="stMetric"] {background:#fff;border:1px solid #e5eaf2;
padding:1rem;border-radius:14px;box-shadow:0 3px 12px rgba(24,49,83,.05);}
.note {color:#64748b;font-size:.92rem;margin-bottom:.8rem;}
</style>
""", unsafe_allow_html=True)

# 중요: GitHub의 CSV 파일을 실제로 업로드한 뒤 Raw 주소를 여기에 넣으세요.
DATA_URL = "https://raw.githubusercontent.com/dltnqls7/dsproject__/refs/heads/main/subject101_preprocessed_1sec.csv"

ACTIVITY_KOREAN = {
    "lying": "누워 있기",
    "sitting": "앉아 있기",
    "standing": "서 있기",
    "walking": "걷기",
    "running": "달리기",
    "cycling": "자전거 타기",
    "nordic walking": "노르딕 워킹",
    "ascending stairs": "계단 올라가기",
    "descending stairs": "계단 내려가기",
    "vacuum cleaning": "청소기 돌리기",
    "ironing": "다림질하기",
    "rope jumping": "줄넘기",
    "lying down": "누워 있기",
    "sitting down": "앉아 있기",
    "standing still": "서 있기",
}
INTENSITY_KOREAN = {
    "low": "낮음", "light": "낮음",
    "medium": "보통", "moderate": "보통",
    "high": "높음", "vigorous": "높음",
    "낮음": "낮음", "보통": "보통", "높음": "높음"
}

@st.cache_data
def load_from_url(url):
    return pd.read_csv(url)

def translate_activity(value):
    original = str(value).strip()
    key = " ".join(original.lower().replace("_", " ").replace("-", " ").split())
    return ACTIVITY_KOREAN.get(key, original)

st.markdown('<div class="title">❤️ 운동 강도와 심박수 분석</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">센서 데이터로 살펴보는 신체 활동과 심박수의 관계</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="question"><b>🔎 핵심 질문</b><br>운동 강도가 높아질수록 심박수도 높아지는가?</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# 데이터 불러오기: URL을 우선 시도하고 실패하면 CSV 업로드 제공
# --------------------------------------------------
st.sidebar.title("📂 데이터 설정")
st.sidebar.caption("GitHub CSV를 읽지 못하면 전처리된 CSV 파일을 직접 업로드하세요.")
uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])

df = None
load_error = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        load_error = f"업로드한 CSV를 읽지 못했습니다: {e}"
else:
    try:
        df = load_from_url(DATA_URL)
        # CSV가 아닌 main.py 같은 텍스트 파일을 잘못 불러온 경우 감지
        required = {"time_seconds", "activity_name", "heart_rate_bpm", "MET", "intensity"}
        if not required.issubset(set(df.columns)):
            df = None
            load_error = (
                "GitHub 주소에서 예상한 CSV 열을 찾지 못했습니다. "
                "저장소의 CSV 파일이 실제 데이터인지 확인하거나 왼쪽 메뉴에서 CSV를 업로드하세요."
            )
    except Exception as e:
        load_error = f"GitHub에서 CSV를 불러오지 못했습니다: {e}"
