
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# ==========================================
# 1. 페이지 설정
# ==========================================
st.set_page_config(
    page_title="운동 강도와 심박수 분석",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ 운동 강도와 심박수 머신러닝 분석")
st.write(
    "PAMAP2 신체 활동 데이터를 이용해 "
    "운동 강도와 심박수의 관계를 분석합니다."
)

st.info(
    "회귀 모델은 MET로 심박수를 예측하고, "
    "분류 모델은 심박수와 움직임 센서값으로 "
    "운동 강도를 구분합니다."
)

# ==========================================
# 2. 활동별 MET 기준
# ==========================================
MET_MAP = {
    1: 1.0,    # 누워 있기
    2: 1.8,    # 앉기
    3: 1.8,    # 서 있기
    4: 3.55,   # 걷기: 3.3~3.8
    5: 7.5,    # 달리기: 7~8
    6: 4.0,    # 자전거 타기
    7: 5.5,    # 노르딕 워킹: 5~6
    9: 1.3,    # TV 시청: 분석용 근사값
    10: 1.8,   # 컴퓨터 작업: 분석용 근사값
    11: 2.0,   # 자동차 운전: 분석용 근사값
    12: 8.0,   # 계단 올라가기
    13: 3.0,   # 계단 내려가기
    16: 3.5,   # 진공청소
    17: 2.3,   # 다림질
    18: 2.0,   # 빨래 개기: 분석용 근사값
    19: 3.0,   # 집 청소: 분석용 근사값
    20: 7.0,   # 축구: 분석용 근사값
    24: 9.0,   # 줄넘기: 8~10
}

ACTIVITY_NAMES = {
    1: "누워 있기",
    2: "앉기",
    3: "서 있기",
    4: "걷기",
    5: "달리기",
    6: "자전거 타기",
    7: "노르딕 워킹",
    9: "TV 시청",
    10: "컴퓨터 작업",
    11: "자동차 운전",
    12: "계단 올라가기",
    13: "계단 내려가기",
    16: "진공청소",
    17: "다림질",
    18: "빨래 개기",
    19: "집 청소",
    20: "축구",
    24: "줄넘기",
}

def classify_intensity(met):
    if met < 3:
        return "낮음"
    elif met < 6:
        return "중간"
    return "높음"


# ==========================================
# 3. 데이터 업로드
# ==========================================
st.header("1. 데이터 준비")

uploaded_file = st.file_uploader(
    "subject101.dat 파일을 업로드하세요.",
    type=["dat", "txt"]
)

if uploaded_file is None:
    st.warning(
        "PAMAP2의 subject101.dat 파일을 업로드하면 "
        "데이터 분석과 두 머신러닝 모델을 실행할 수 있습니다."
    )
    st.markdown(
        "데이터 출처: [PAMAP2 공식 데이터셋]"
        "(https://archive.ics.uci.edu/dataset/231/"
        "pamap2+physical+activity+monitoring)"
    )
    st.stop()

try:
    raw = pd.read_csv(
        uploaded_file,
        sep=r"\s+",
        header=None,
        na_values=["NaN", "nan"]
    )

    if raw.shape[1] != 54:
        st.error(
            f"54개 열이 필요하지만 {raw.shape[1]}개 열이 발견됐습니다. "
            "PAMAP2의 원본 .dat 파일인지 확인하세요."
        )
        st.stop()

    raw = raw.apply(pd.to_numeric, errors="coerce")

except Exception as e:
    st.error(f"파일을 읽지 못했습니다: {e}")
    st.stop()

st.success("데이터 파일을 읽었습니다.")

# ==========================================
# 4. 전처리
# ==========================================
# PAMAP2 원본 열 번호는 1부터 시작함.
# Python에서는 0부터 시작하므로 인덱스가 하나씩 작아짐.

data = pd.DataFrame({
    "시간": raw[0],
    "활동코드": raw[1],
    "심박수": raw[2],
})

# 각 센서의 첫 번째 3축 가속도 사용
sensor_columns = {
    "손목가속도": [4, 5, 6],
    "가슴가속도": [21, 22, 23],
    "발목가속도": [38, 39, 40],
}

for name, columns in sensor_columns.items():
    data[name] = np.sqrt(
        (raw[columns] ** 2).sum(axis=1, min_count=3)
    )

data["MET"] = data["활동코드"].map(MET_MAP)
data["활동명"] = data["활동코드"].map(ACTIVITY_NAMES)
data["강도"] = data["MET"].apply(
    lambda x: classify_intensity(x) if pd.notna(x) else np.nan
)

# 활동 코드 0은 활동 사이의 전환 구간이므로 제외
# 심박수 및 가속도 결측값도 제거
required_columns = [
    "시간", "활동코드", "심박수", "MET",
    "손목가속도", "가슴가속도", "발목가속도"
]

clean = data[
    (data["활동코드"] != 0) &
    data["활동명"].notna()
].dropna(subset=required_columns).copy()

clean = clean[
    np.isfinite(clean[required_columns].select_dtypes(
        include=[np.number]
    )).all(axis=1)
]

clean["활동코드"] = clean["활동코드"].astype(int)

if len(clean) < 30:
    st.error("분석에 사용할 데이터가 너무 적습니다.")
    st.stop()

# ==========================================
# 5. 데이터 요약
# ==========================================
st.header("2. 데이터 살펴보기")

col1, col2, col3, col4 = st.columns(4)

col1.metric("원본 행 수", f"{len(raw):,}")
col2.metric("분석 행 수", f"{len(clean):,}")
col3.metric("활동 종류", f"{clean['활동코드'].nunique()}종")
col4.metric("평균 심박수", f"{clean['심박수'].mean():.1f} BPM")

with st.expander("분석에 사용한 데이터 미리 보기"):
    st.dataframe(clean.head(100), use_container_width=True)

st.subheader("활동별 평균 심박수")

activity_summary = (
    clean.groupby(["활동코드", "활동명", "MET"], as_index=False)
    .agg(
        평균심박수=("심박수", "mean"),
        측정개수=("심박수", "size")
    )
    .sort_values("MET")
)

st.dataframe(
    activity_summary.style.format({
        "MET": "{:.2f}",
        "평균심박수": "{:.1f}",
        "측정개수": "{:,.0f}"
    }),
    use_container_width=True
)

st.bar_chart(
    activity_summary.set_index("활동명")[["평균심박수"]]
)

st.subheader("MET와 심박수의 관계")

st.scatter_chart(
    clean,
    x="MET",
    y="심박수",
    color="강도"
)

st.caption(
    "MET는 활동별 대표값이며, 같은 활동에 속한 행에는 "
    "동일한 MET가 부여됩니다."
)

# ==========================================
# 6. 회귀 모델
# ==========================================
st.header("3. 모델 1 — 회귀: 심박수 예측")

st.write(
    "입력값: MET / 예측값: 심박수(BPM)"
)

reg_data = clean[["MET", "심박수", "활동코드"]].copy()

if reg_data["활동코드"].nunique() < 2:
    st.warning("회귀 모델을 학습하려면 여러 활동 종류가 필요합니다.")
else:
    # 동일 활동의 연속 측정값이 학습·테스트 양쪽에 섞이지 않도록
    # 활동 코드 단위로 나눕니다.
    group_ids = np.array(sorted(reg_data["활동코드"].unique()))
    rng = np.random.default_rng(42)
    rng.shuffle(group_ids)

    n_test_groups = max(1, round(len(group_ids) * 0.25))
    test_groups = group_ids[:n_test_groups]

    reg_train = reg_data[~reg_data["활동코드"].isin(test_groups)]
    reg_test = reg_data[reg_data["활동코드"].isin(test_groups)]

    if len(reg_train) < 2 or len(reg_test) < 2:
        st.warning("활동 종류가 부족해 회귀 모델을 평가할 수 없습니다.")
    else:
        reg_model = LinearRegression()
        reg_model.fit(reg_train[["MET"]], reg_train["심박수"])

        reg_pred = reg_model.predict(reg_test[["MET"]])

        mae = mean_absolute_error(reg_test["심박수"], reg_pred)
        rmse = np.sqrt(
            mean_squared_error(reg_test["심박수"], reg_pred)
        )
        r2 = r2_score(reg_test["심박수"], reg_pred)

        a, b, c = st.columns(3)
        a.metric("MAE", f"{mae:.2f} BPM")
        b.metric("RMSE", f"{rmse:.2f} BPM")
        c.metric("R²", f"{r2:.3f}")

        st.write(
            f"학습된 회귀식: 심박수 = "
            f"{reg_model.coef_[0]:.2f} × MET "
            f"+ {reg_model.intercept_:.2f}"
        )

        reg_result = pd.DataFrame({
            "실제 심박수": reg_test["심박수"].to_numpy(),
            "예측 심박수": reg_pred
        })

        st.scatter_chart(
            reg_result,
            x="실제 심박수",
            y="예측 심박수"
        )

        with st.expander("새로운 MET 값으로 심박수 예측하기"):
            met_input = st.slider(
                "운동 강도 (MET)",
                min_value=1.0,
                max_value=10.0,
                value=5.0,
                step=0.1
            )

            predicted_hr = reg_model.predict(
                pd.DataFrame({"MET": [met_input]})
            )[0]

            st.metric("모델 예측 심박수", f"{predicted_hr:.1f} BPM")
            st.caption(
                "예측값은 이 데이터셋에 맞춘 통계적 추정치이며 "
                "개인의 실제 심박수나 안전 기준을 의미하지 않습니다."
            )

# ==========================================
# 7. 분류 모델
# ==========================================
st.header("4. 모델 2 — 분류: 운동 강도 구분")

st.write(
    "입력값: 심박수와 손목·가슴·발목 가속도 크기 / "
    "정답: MET 기준으로 정의한 운동 강도"
)

features = [
    "심박수",
    "손목가속도",
    "가슴가속도",
    "발목가속도"
]

X = clean[features]
y = clean["강도"]
groups = clean["활동코드"]

# 활동 종류를 기준으로 테스트 데이터를 분리
unique_groups = np.array(sorted(groups.unique()))
rng = np.random.default_rng(42)
rng.shuffle(unique_groups)

n_test = max(1, round(len(unique_groups) * 0.25))
test_groups = unique_groups[:n_test]

train_mask = ~groups.isin(test_groups)
test_mask = groups.isin(test_groups)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

if (
    len(X_train) < 10 or
    len(X_test) < 10 or
    y_train.nunique() < 2
):
    st.warning(
        "활동 종류 또는 강도 범주가 부족합니다. "
        "더 많은 참가자의 데이터를 합쳐 분석해 보세요."
    )
else:
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=3,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    st.metric("분류 정확도", f"{accuracy:.1%}")

    labels = ["낮음", "중간", "높음"]

    cm = confusion_matrix(
        y_test,
        pred,
        labels=labels
    )

    st.subheader("혼동 행렬")

    st.dataframe(
        pd.DataFrame(
            cm,
            index=[f"실제 {x}" for x in labels],
            columns=[f"예측 {x}" for x in labels]
        ),
        use_container_width=True
    )

    st.subheader("분류 성능 보고서")

    report = classification_report(
        y_test,
        pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    st.dataframe(
        pd.DataFrame(report).T,
        use_container_width=True
    )

    st.subheader("변수 중요도")

    importance = pd.DataFrame({
        "변수": features,
        "중요도": clf.feature_importances_
    }).sort_values("중요도", ascending=False)

    st.bar_chart(
        importance.set_index("변수")
    )

    with st.expander("새로운 측정값으로 강도 분류하기"):
        hr_input = st.number_input(
            "심박수 (BPM)", 30.0, 220.0, 100.0
        )
        hand_input = st.number_input(
            "손목 가속도 크기", 0.0, 100.0, 9.8
        )
        chest_input = st.number_input(
            "가슴 가속도 크기", 0.0, 100.0, 9.8
        )
        ankle_input = st.number_input(
            "발목 가속도 크기", 0.0, 100.0, 9.8
        )

        if st.button("운동 강도 분류하기"):
            new_data = pd.DataFrame([{
                "심박수": hr_input,
                "손목가속도": hand_input,
                "가슴가속도": chest_input,
                "발목가속도": ankle_input
            }])

            result = clf.predict(new_data)[0]
            st.success(f"예측한 운동 강도: {result}")

            st.caption(
                "이 분류는 사전에 정한 MET 등급을 학습한 결과이며, "
                "의학적 운동 강도 판정은 아닙니다."
            )

# ==========================================
# 8. 주의사항
# ==========================================
st.divider()
st.header("5. 분석 시 주의할 점")

st.markdown("""
- MET는 실제 각 행에서 측정한 운동 강도가 아니라 활동별 대표값입니다.
- 일부 가사 활동의 MET는 설명서에 값이 없어서 분석용 근사값으로 설정했습니다.
- 낮음·중간·높음의 경계는 분석을 위해 MET 3과 6으로 정한 것입니다.
- 같은 참가자의 데이터만 사용하면 다른 사람에게 일반화하기 어렵습니다.
- 심박수와 가속도 센서값의 결측 행을 제거했으므로 원본보다 분석 행 수가 적습니다.
- 회귀와 분류 결과는 데이터 분할 방식과 전처리 기준에 따라 달라질 수 있습니다.
""")

st.caption(
    "데이터 출처: PAMAP2 Physical Activity Monitoring Dataset, "
    "UCI Machine Learning Repository."
)
