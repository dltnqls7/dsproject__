
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

# GitHub 원본 데이터
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
    st.error("데이터를 불러오지 못했습니다. GitHub CSV 주소와 저장소 공개 여부를 확인해 주세요.")
    st.exception(e)
    st.stop()

# 필요한 열 확인 및 전처리
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

# 앞서 정한 조건: 80:20, random_state=42
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# 심박수 구간 경계는 학습 데이터에서만 계산
low_cut, high_cut = train_df["heart_rate_bpm"].quantile([1/3, 2/3])

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

# 베이스라인 + 세 후보 모델
models = {
    "베이스라인": DummyClassifier(strategy="most_frequent"),
    "로지스틱 회귀": make_pipeline(
