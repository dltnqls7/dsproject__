
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="운동 강도와 심박수 분석",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ 운동 강도와 심박수의 관계 분석")
st.write(
    "PAMAP2 신체 활동 데이터를 활용하여 시간에 따른 심박수 변화, "
    "활동별 심박수 차이, 운동 강도(MET)와 평균 심박수의 관계를 분석합니다."
)

# CSV 업로드
uploaded_file = st.file_uploader(
    "전처리한 CSV 파일을 업로드하세요.",
    type=["csv"]
)

if uploaded_file is None:
    st.info("subject101_preprocessed_1sec.csv 파일을 업로드하세요.")
    st.stop()

@st.cache_data
def load_data(file_bytes):
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))

df = load_data(uploaded_file.getvalue())

# 필수 열 검사
required_columns = [
    "time_seconds", "activity_name",
    "heart_rate_bpm", "MET"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"CSV에 필요한 열이 없습니다: {missing_columns}")
    st.stop()

# 데이터 정리
for col in ["time_seconds", "heart_rate_bpm", "MET"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=["time_seconds", "activity_name", "heart_rate_bpm", "MET"]
)

if df.empty:
    st.error("분석할 수 있는 데이터가 없습니다.")
    st.stop()

# 요약 수치
c1, c2, c3 = st.columns(3)
c1.metric("분석 데이터 수", f"{len(df):,}개")
c2.metric("평균 심박수", f"{df['heart_rate_bpm'].mean():.1f} BPM")
c3.metric("최대 심박수", f"{df['heart_rate_bpm'].max():.1f} BPM")

st.divider()

# 1. 시간에 따른 심박수
st.header("1. 시간에 따른 심박수 변화")

time_df = (
    df.sort_values("time_seconds")
    [["time_seconds", "heart_rate_bpm"]]
    .set_index("time_seconds")
)

st.line_chart(
    time_df,
    x_label="시간(초)",
    y_label="심박수(BPM)"
)

st.caption("시간에 따라 심박수가 어떻게 변하는지 확인합니다.")

st.divider()

# 2. 활동별 심박수 비교
st.header("2. 활동별 심박수 비교")

activity_order = (
    df.groupby("activity_name")["heart_rate_bpm"]
    .median()
    .sort_values()
    .index.tolist()
)

selected_activities = st.multiselect(
    "비교할 활동을 선택하세요.",
    options=activity_order,
    default=activity_order
)

if selected_activities:
    activity_df = df[
        df["activity_name"].isin(selected_activities)
    ]

    # 활동별 중앙값
    median_hr = (
        activity_df.groupby("activity_name")["heart_rate_bpm"]
        .median()
        .sort_values()
    )

    st.subheader("활동별 중앙값 심박수")
    st.bar_chart(
        median_hr,
        x_label="활동",
        y_label="중앙값 심박수(BPM)"
    )

    # 활동별 평균값
    mean_hr = (
        activity_df.groupby("activity_name")["heart_rate_bpm"]
        .mean()
        .sort_values()
    )

    st.subheader("활동별 평균 심박수")
    st.bar_chart(
        mean_hr,
        x_label="활동",
        y_label="평균 심박수(BPM)"
    )

    # 활동별 분포를 숫자로 확인
    distribution = (
        activity_df.groupby("activity_name")["heart_rate_bpm"]
        .agg(
            평균="mean",
            중앙값="median",
            최솟값="min",
            최댓값="max",
            데이터수="count"
        )
        .round(2)
    )

    st.subheader("활동별 심박수 통계")
    st.dataframe(distribution, use_container_width=True)

else:
    st.warning("활동을 하나 이상 선택하세요.")

st.caption(
    "막대그래프는 활동별 평균과 중앙값을 비교합니다. "
    "상자그림과 달리 분포의 사분위 범위는 표시하지 않습니다."
)

st.divider()

# 3. MET와 평균 심박수
st.header("3. 운동 강도(MET)와 평균 심박수의 관계")

group_columns = ["activity_name", "MET"]
if "activity_id" in df.columns:
    group_columns.insert(0, "activity_id")

summary = (
    df.groupby(group_columns, as_index=False)
    .agg(
        mean_hr=("heart_rate_bpm", "mean"),
        median_hr=("heart_rate_bpm", "median"),
        data_count=("heart_rate_bpm", "size")
    )
    .sort_values("MET")
)

st.subheader("활동별 평균 심박수")

st.dataframe(
    summary.rename(columns={
        "activity_id": "활동 코드",
        "activity_name": "활동",
        "MET": "MET 추정값",
        "mean_hr": "평균 심박수(BPM)",
        "median_hr": "중앙값 심박수(BPM)",
        "data_count": "데이터 수"
    }).round(2),
    use_container_width=True,
    hide_index=True
)

# Streamlit 기본 산점도
st.subheader("MET별 평균 심박수 산점도")

scatter_df = summary[
    ["MET", "mean_hr", "activity_name"]
].rename(columns={
    "MET": "운동 강도(MET)",
    "mean_hr": "평균 심박수(BPM)",
    "activity_name": "활동"
})

st.scatter_chart(
    scatter_df,
    x="운동 강도(MET)",
    y="평균 심박수(BPM)",
    color="활동"
)

st.caption(
    "각 점은 활동별 평균을 나타냅니다. MET는 활동별 참고값을 연결한 추정치입니다."
)

st.divider()

# 4. 결론 및 주의사항
st.header("4. 분석 시 주의사항")

st.markdown("""
- 분석 대상은 참가자 101의 기록이므로 모든 사람에게 결과를 일반화할 수 없습니다.
- MET는 원본에서 직접 측정한 값이 아니라 활동별 참고값을 연결한 것입니다.
- 그래프만으로 인과관계나 머신러닝 예측 정확도가 입증되는 것은 아닙니다.
- 심박수 예측 성능을 확인하려면 별도의 회귀 모델 평가가 필요합니다.
""")

st.success("시각화가 완료되었습니다.")
