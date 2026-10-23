
import streamlit as st
import pandas as pd

# 페이지 설정
st.set_page_config(
    page_title="운동 강도와 심박수 분석",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ 운동 강도와 심박수의 관계 분석")
st.write(
    "PAMAP2 신체 활동 데이터를 활용하여 "
    "시간에 따른 심박수 변화와 활동별 심박수 분포, "
    "운동 강도(MET)와 평균 심박수의 관계를 분석합니다."
)

# CSV 파일 불러오기
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

uploaded_file = st.file_uploader(
    "전처리한 CSV 파일을 업로드하세요.",
    type=["csv"]
)

if uploaded_file is None:
    st.info("subject101_preprocessed_1sec.csv 파일을 업로드하면 분석 결과가 표시됩니다.")
    st.stop()

df = load_data(uploaded_file)

# 필수 열 확인
required_columns = [
    "time_seconds",
    "activity_name",
    "heart_rate_bpm",
    "MET"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"CSV 파일에 필요한 열이 없습니다: {missing_columns}")
    st.stop()

# 숫자형 변환 및 결측치 제거
df["time_seconds"] = pd.to_numeric(
    df["time_seconds"], errors="coerce"
)
df["heart_rate_bpm"] = pd.to_numeric(
    df["heart_rate_bpm"], errors="coerce"
)
df["MET"] = pd.to_numeric(df["MET"], errors="coerce")

df = df.dropna(
    subset=["time_seconds", "heart_rate_bpm", "MET", "activity_name"]
)

if df.empty:
    st.error("분석할 수 있는 데이터가 없습니다.")
    st.stop()

# 요약 지표
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("분석 데이터 수", f"{len(df):,}개")

with col2:
    st.metric("평균 심박수", f"{df['heart_rate_bpm'].mean():.1f} BPM")

with col3:
    st.metric("최대 심박수", f"{df['heart_rate_bpm'].max():.1f} BPM")

st.divider()

# 그래프 1: 시간에 따른 심박수
st.header("1. 시간에 따른 심박수 변화")

time_df = df.sort_values("time_seconds").set_index("time_seconds")

st.line_chart(
    time_df[["heart_rate_bpm"]],
    x_label="시간(초)",
    y_label="심박수(BPM)"
)

st.caption(
    "시간에 따라 심박수가 어떻게 변하는지 확인합니다. "
    "급격한 변화는 실제 활동과 측정 구간을 함께 살펴봐야 합니다."
)

st.divider()

# 그래프 2: 활동별 심박수 분포
st.header("2. 활동별 심박수 분포")

activity_order = (
    df.groupby("activity_name")["heart_rate_bpm"]
    .median()
    .sort_values()
    .index.tolist()
)

selected_activities = st.multiselect(
    "비교할 활동 선택",
    options=activity_order,
    default=activity_order
)

if selected_activities:
    activity_df = df[
        df["activity_name"].isin(selected_activities)
    ]

    chart_data = activity_df.pivot(
        columns="activity_name",
        values="heart_rate_bpm"
    )

    st.box_chart(
        chart_data,
        x_label="활동",
        y_label="심박수(BPM)"
    )
else:
    st.warning("비교할 활동을 하나 이상 선택하세요.")

st.caption(
    "상자그림을 통해 활동별 심박수의 중앙값과 분포 차이를 비교합니다."
)

st.divider()

# 그래프 3: MET와 평균 심박수
st.header("3. 운동 강도(MET)와 평균 심박수의 관계")

summary = (
    df.groupby(["activity_id", "activity_name", "MET"], as_index=False)
    .agg(
        mean_hr=("heart_rate_bpm", "mean"),
        median_hr=("heart_rate_bpm", "median"),
        data_count=("heart_rate_bpm", "size")
    )
    .sort_values("MET")
)

st.scatter_chart(
    summary,
    x="MET",
    y="mean_hr",
    color="activity_name",
    size="data_count",
    x_label="운동 강도(MET)",
    y_label="평균 심박수(BPM)"
)

st.dataframe(
    summary.rename(columns={
        "activity_name": "활동",
        "MET": "MET 추정값",
        "mean_hr": "평균 심박수(BPM)",
        "median_hr": "중앙값 심박수(BPM)",
        "data_count": "데이터 수"
    }).round(2),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "MET는 활동별 참고값을 연결한 추정치입니다. "
    "점 하나는 하나의 활동을 나타냅니다."
)

st.divider()

# 결론 및 주의사항
st.header("4. 분석 시 주의사항")

st.markdown("""
- 이 데이터는 참가자 101의 기록이므로 모든 사람에게 결과를 일반화할 수 없습니다.
- MET는 원본에서 직접 측정한 값이 아니라 활동별 참고값을 연결한 것입니다.
- 그래프에서 관계가 보이더라도 그것만으로 인과관계나 예측 정확도가 입증되는 것은 아닙니다.
- 심박수 예측 성능은 별도의 회귀 모델로 평가해야 합니다.
""")

st.success("시각화가 완료되었습니다.")
