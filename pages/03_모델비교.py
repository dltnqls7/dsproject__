
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# --------------------------------------------------
# 1. 페이지 설정
# --------------------------------------------------

st.set_page_config(
    page_title="모델 3종 비교",
    page_icon="📊",
    layout="wide"
)

st.title("📊 심박수 예측 모델 비교")

st.write(
    "운동 강도(MET)를 이용해 심박수 구간을 예측하고 "
    "로지스틱 회귀, 결정 트리, 랜덤 포레스트의 성능을 비교합니다."
)

st.divider()


# --------------------------------------------------
# 2. GitHub 데이터 불러오기
# --------------------------------------------------

DATA_URL = (
    "https://raw.githubusercontent.com/dltnqls7/"
    "dsproject__/refs/heads/main/subject101_preprocessed.csv"
)


@st.cache_data
def load_data(url):
    return pd.read_csv(url)


try:
    raw_df = load_data(DATA_URL)

except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.write("GitHub CSV 주소와 저장소 공개 여부를 확인해 주세요.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# 3. 데이터 전처리
# --------------------------------------------------

required = ["MET", "heart_rate_bpm"]

if not all(col in raw_df.columns for col in required):
    st.error("CSV에 MET 또는 heart_rate_bpm 열이 없습니다.")
    st.stop()

df = raw_df[required].copy()

for col in required:
    df[col] = pd.to_numeric(df[col], errors="coerce")

before = len(df)

df = df.dropna(subset=required)

removed = before - len(df)

if len(df) < 20:
    st.error("학습에 사용할 데이터가 충분하지 않습니다.")
    st.stop()


# --------------------------------------------------
# 4. 데이터 정보
# --------------------------------------------------

st.subheader("1. 데이터 확인")

col1, col2, col3 = st.columns(3)

col1.metric("전체 데이터", f"{len(raw_df):,}개")
col2.metric("전처리 후 데이터", f"{len(df):,}개")
col3.metric("제거된 데이터", f"{removed:,}개")

st.caption(
    "MET와 심박수 열을 숫자형으로 변환하고, "
    "두 열 중 결측값이 있는 행을 제거했습니다."
)

with st.expander("전처리된 데이터 확인"):
    st.dataframe(df.head(10), use_container_width=True)


# --------------------------------------------------
# 5. 학습 데이터와 테스트 데이터 분리
# --------------------------------------------------

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

st.subheader("2. 학습 데이터와 테스트 데이터")

col1, col2 = st.columns(2)

col1.metric("학습 데이터", f"{len(train_df):,}개")
col2.metric("테스트 데이터", f"{len(test_df):,}개")

st.write(
    "전체 데이터를 학습용 80%, 테스트용 20%로 나누었습니다. "
    "random_state=42를 사용하여 동일한 조건에서 비교할 수 있도록 했습니다."
)


# --------------------------------------------------
# 6. 심박수 구간 분류
# --------------------------------------------------

low_cut, high_cut = train_df["heart_rate_bpm"].quantile(
    [1 / 3, 2 / 3]
)

classes = ["낮음", "보통", "높음"]


def to_class(values):
    return pd.cut(
        values,
        bins=[-np.inf, low_cut, high_cut, np.inf],
        labels=classes,
        include_lowest=True
    )


X_train = train_df[["MET"]]
X_test = test_df[["MET"]]

y_train = to_class(train_df["heart_rate_bpm"])
y_test = to_class(test_df["heart_rate_bpm"])

if y_train.isna().any() or y_test.isna().any():
    st.error("심박수 구간을 나누는 과정에서 문제가 발생했습니다.")
    st.stop()

st.write(
    f"심박수 구간 기준: 낮음 ≤ {low_cut:.1f} BPM, "
    f"보통 ≤ {high_cut:.1f} BPM, 그 초과는 높음"
)

st.caption(
    "심박수 구간의 경계는 학습 데이터에서만 계산하여 "
    "테스트 데이터 정보가 학습에 미리 반영되지 않도록 했습니다."
)


# --------------------------------------------------
# 7. 모델 생성
# --------------------------------------------------

models = {
    "베이스라인": DummyClassifier(
        strategy="most_frequent"
    ),

    "로지스틱 회귀": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
    ),

    "결정 트리": DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=4,
        min_samples_split=30,
        min_samples_leaf=15,
        random_state=42
    ),

    "랜덤 포레스트": RandomForestClassifier(
        class_weight="balanced",
        n_estimators=200,
        max_depth=5,
        min_samples_split=30,
        min_samples_leaf=15,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
}


# --------------------------------------------------
# 8. 모델 학습 및 성능 평가
# --------------------------------------------------

results = []
trained_models = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(
        y_train,
        train_pred
    )

    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )

    precision = precision_score(
        y_test,
        test_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        test_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        test_pred,
        average="weighted",
        zero_division=0
    )

    results.append({
        "모델": name,
        "학습 정확도": train_accuracy,
        "테스트 정확도": test_accuracy,
        "정밀도": precision,
        "재현율": recall,
        "F1 점수": f1,
        "과적합 차이": train_accuracy - test_accuracy
    })

    trained_models[name] = model


results_df = pd.DataFrame(results)


# --------------------------------------------------
# 9. 모델 성능 비교표
# --------------------------------------------------

st.divider()

st.subheader("3. 모델별 성능 비교")

display_df = results_df.copy()

metric_columns = [
    "학습 정확도",
    "테스트 정확도",
    "정밀도",
    "재현율",
    "F1 점수",
    "과적합 차이"
]

for col in metric_columns:
    display_df[col] = display_df[col].map(
        lambda x: f"{x * 100:.1f}%"
    )

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "정밀도, 재현율, F1 점수는 클래스별 성능을 가중 평균한 값입니다."
)


# --------------------------------------------------
# 10. 정확도 비교 그래프
# --------------------------------------------------

st.subheader("4. 모델별 정확도 비교")

accuracy_chart = results_df[
    ["모델", "테스트 정확도"]
].set_index("모델")

st.bar_chart(accuracy_chart)

st.caption(
    "테스트 정확도가 높을수록 처음 보는 데이터에서 "
    "심박수 구간을 더 정확하게 예측한 것입니다."
)


# --------------------------------------------------
# 11. F1 점수 비교 그래프
# --------------------------------------------------

st.subheader("5. 모델별 F1 점수 비교")

f1_chart = results_df[
    ["모델", "F1 점수"]
].set_index("모델")

st.bar_chart(f1_chart)

st.caption(
    "F1 점수는 정밀도와 재현율을 함께 고려한 지표입니다."
)


# --------------------------------------------------
# 12. 과적합 확인
# --------------------------------------------------

st.subheader("6. 과적합 확인")

gap_df = results_df[
    ["모델", "학습 정확도", "테스트 정확도", "과적합 차이"]
].copy()

for col in [
    "학습 정확도",
    "테스트 정확도",
    "과적합 차이"
]:
    gap_df[col] = gap_df[col].map(
        lambda x: f"{x * 100:.1f}%"
    )

st.dataframe(
    gap_df,
    use_container_width=True,
    hide_index=True
)

st.write(
    "학습 정확도와 테스트 정확도의 차이가 클수록 "
    "과적합 가능성이 있습니다."
)


# --------------------------------------------------
# 13. 혼동 행렬
# --------------------------------------------------

st.subheader("7. 혼동 행렬 확인")

selected_model = st.selectbox(
    "확인할 모델을 선택하세요.",
    list(trained_models.keys())
)

selected_prediction = trained_models[selected_model].predict(
    X_test
)

cm = confusion_matrix(
    y_test,
    selected_prediction,
    labels=classes
)

cm_df = pd.DataFrame(
    cm,
    index=[f"실제 {c}" for c in classes],
    columns=[f"예측 {c}" for c in classes]
)

st.write(f"선택한 모델: **{selected_model}**")

st.dataframe(
    cm_df,
    use_container_width=True
)

st.caption(
    "행은 실제 심박수 구간, 열은 모델이 예측한 심박수 구간입니다. "
    "대각선에 있는 숫자가 클수록 정확하게 분류한 경우가 많습니다."
)


# --------------------------------------------------
# 14. 모델 원리와 주요 설정
# --------------------------------------------------

st.divider()

st.subheader("8. 모델 원리와 주요 설정")

with st.expander("베이스라인"):
    st.write(
        "가장 많이 등장하는 심박수 구간만 계속 예측하는 기본 모델입니다. "
        "다른 모델의 성능이 이 기준보다 나은지 확인하기 위해 사용합니다."
    )

with st.expander("로지스틱 회귀"):
    st.write(
        "입력된 운동 강도(MET)를 바탕으로 각 심박수 구간에 속할 "
        "확률을 계산하는 분류 모델입니다."
    )
    st.write(
        "- StandardScaler: 데이터의 크기를 비슷한 범위로 조정"
    )
    st.write(
        "- class_weight='balanced': 적은 클래스의 중요도를 높임"
    )
    st.write(
        "- max_iter=1000: 최적의 결과를 찾기 위한 반복 횟수"
    )

with st.expander("결정 트리"):
    st.write(
        "조건을 여러 번 나누면서 심박수 구간을 예측하는 모델입니다."
    )
    st.write(
        "- max_depth=4: 나무의 최대 깊이를 제한"
    )
    st.write(
        "- min_samples_split=30: 분기하기 위한 최소 데이터 수"
    )
    st.write(
        "- min_samples_leaf=15: 마지막 단계에 남겨야 하는 최소 데이터 수"
    )
    st.write(
        "- class_weight='balanced': 클래스별 가중치 조정"
    )

with st.expander("랜덤 포레스트"):
    st.write(
        "여러 개의 결정 트리를 만들고, 각 트리의 예측 결과를 "
        "종합하여 최종 심박수 구간을 예측합니다."
    )
    st.write(
        "- n_estimators=200: 결정 트리 200개 사용"
    )
    st.write(
        "- max_depth=5: 각 트리의 최대 깊이 제한"
    )
    st.write(
        "- min_samples_split=30: 분기하기 위한 최소 데이터 수"
    )
    st.write(
        "- min_samples_leaf=15: 마지막 단계에 남겨야 하는 최소 데이터 수"
    )
    st.write(
        "- max_features='sqrt': 분기할 때 고려하는 특성 수 제한"
    )
    st.write(
        "- class_weight='balanced': 클래스별 가중치 조정"
    )


# --------------------------------------------------
# 15. 평가 지표 설명
# --------------------------------------------------

st.subheader("9. 평가 지표 설명")

st.markdown("""
- **정확도(Accuracy)**: 전체 예측 중 맞힌 비율
- **정밀도(Precision)**: 특정 구간이라고 예측한 것 중 실제로 맞힌 비율
- **재현율(Recall)**: 실제 특정 구간에 속한 데이터 중 찾아낸 비율
- **F1 점수(F1-score)**: 정밀도와 재현율을 함께 고려한 점수
""")

st.caption(
    "이번 실험은 심박수를 낮음, 보통, 높음의 세 구간으로 나누었으므로 "
    "분류 모델에 사용하는 평가 지표를 적용했습니다."
)


# --------------------------------------------------
# 16. 최종 결과 요약
# --------------------------------------------------

st.divider()

st.subheader("10. 최종 결과")

best_model = results_df.loc[
    results_df["테스트 정확도"].idxmax()
]

st.success(
    f"테스트 정확도 기준 가장 높은 모델은 "
    f"'{best_model['모델']}'입니다."
)

st.write(
    f"테스트 정확도: {best_model['테스트 정확도'] * 100:.1f}%"
)

st.write(
    f"F1 점수: {best_model['F1 점수'] * 100:.1f}%"
)

st.write(
    "이번 실험에서는 운동 강도(MET)를 입력하여 "
    "심박수 구간을 예측하고 네 가지 모델의 성능을 비교했습니다. "
    "다만 이 결과만으로 운동 강도가 심박수 변화의 원인이라고 "
    "단정할 수는 없습니다."
)
