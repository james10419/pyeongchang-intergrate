# 평창군 학교 통폐합 제언 앱 (Pyeongchang school intergration helper)

평창군 학교별 인구수, 위치 정보를 읽고 해당 년도의 학교별 통폐합 가능여부를 출력하는 프로그램입니다. 

통폐합은 학생수 및 통폐합시 학교간 통학거리를 고려하여 진행합니다. 거리는 위도와 경도 정보를 이용하여 유클리드 거리로 계산합니다.

현재 자료에서는 2022-2024년도 평창군 내 학교별 학생수가 정리되어 있습니다.


## 프로그램 구동 원리

<img width="960" height="586" alt="image" src="https://github.com/user-attachments/assets/78450ea4-7d8d-413d-b813-31f2da51116c" />

### 1. 시작 화면

시작 화면에서는 분석할 엑셀 파일을 선택하고 각 열의 이름을 지정합니다.

학생수 열에는 분석하고 싶은 연도를 함께 적습니다. (ex. 2024년도 학생수)

통폐합 기준을 설정할 수 있습니다. 통폐합 시 최대 통학거리와 최소 학생수를 지정하여 통폐합 시 발생할 수 있는 문제에 대해 제언합니다.

### 2. 입력 단계

<img width="960" height="580" alt="image" src="https://github.com/user-attachments/assets/d93813be-05e2-432b-9cb3-50ac7066ed9b" />

통폐합 기준과 각 열의 정보를 입력하였다면, 분석 실행 버튼을 클릭하여 데이터 분석을 실행합니다.

엑셀 파일 내 학교들 중 기준에 적합하지 않은 학교들을 골라내어 결과 창에 출력합니다. 

결과창을 확인하면 어느 학교가 어떤 이유로 통폐합이 불가능한지 출력됩니다. 

## 결론
본 프로젝트를 통해 데이터 분석 기법에 대해 학습할 수 있었고, 직접 Policy를 지정하여 분류 모델을 제작하는 경험을 해볼 수 있었습니다.

이후, ARIMA 등 시계열 분석 모델을 이용하여 미래 학생수 변화를 예측하여 학교 통폐합 제언 모델을 업그레이드 할 수도 있습니다.


-----
# Pyeongchang School Integration Helper
This program analyzes student population and location data for schools in Pyeongchang County and outputs recommendations on possible school consolidations for a selected year.

Consolidation decisions are based on student enrollment numbers and the commuting distance required if schools are merged. 
Distances are calculated using latitude and longitude through Euclidean distance.

The current dataset includes student counts for schools in Pyeongchang County from 2022 to 2024.

## How the Program Works
<img width="960" height="586" alt="image" src="https://github.com/user-attachments/assets/78450ea4-7d8d-413d-b813-31f2da51116c" />

### 1. Start Screen

The start screen allows users to:

Select an Excel file containing the school data

Specify the names of each relevant column

Choose the student count column for the year to be analyzed (e.g., “2024 Student Count”)

Set consolidation criteria such as:

Minimum number of students

Maximum allowed commuting distance after consolidation

These settings help evaluate potential issues and guide suggestions during the analysis.

### 2. Input Phase
<img width="960" height="580" alt="image" src="https://github.com/user-attachments/assets/d93813be-05e2-432b-9cb3-50ac7066ed9b" />

After configuring the consolidation criteria and specifying the column names, clicking Run Analysis performs the data evaluation.

The program identifies schools that do not meet the required standards and displays them in the results window.

From the output, users can observe:

Which schools are candidates for consolidation

Which schools cannot be consolidated and the reasons why

Detailed information for each school based on the configured criteria

# Conclusion

Through this project, I gained hands-on experience with data analysis techniques and learned how to design a rule-based classification model using custom-defined policies.

In the future, the system can be upgraded using time-series forecasting models such as ARIMA, allowing predictions of future student population trends and enabling more advanced school consolidation recommendations.
