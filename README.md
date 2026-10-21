# dsproject__
# 운동 강도와 심박수는 정말 관련이 있는가?
## 팀원
- 20321 이수빈, 20618 유성우
## 질문
"운동 강도와 심박수는 정말 관련이 있는가?"
## 왜 이 주제인가 (팀이 씁니다)
운동쪽 진로와 관련되어 있고, 운동 강도와 심박수의 관련이 당연해 보이는 사실을 확인하는 데서 끝나는 것이 아니라, 실제 데이터로 관계의 정도를 분석하고 예측 성능까지 검증해 보고 싶었기 때문입니다.
## 데이터
- 출처: UCI Machine Learning Repository, PAMAP2 Physical Activity Monitoring Dataset. 여러 신체 활동 중 측정한 심박수와 신체 움직임 센서 데이터
- 기간·행 수·열 수: 참가자 101의 데이터 기준 약 376,417행 · 54열
- 사용한 열: 시간(time), 활동 코드(activity ID), 심박수(heart rate), 가속도 센서 데이터(acceleration). 운동 강도(MET)는 활동 코드에 대응하는 값을 연결하여 활용
- 개인 단위 정보 없음 확인: 이름·연락처·주소·학번 등 직접적인 개인 식별 정보 없음.
- 데이터 주소: https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
