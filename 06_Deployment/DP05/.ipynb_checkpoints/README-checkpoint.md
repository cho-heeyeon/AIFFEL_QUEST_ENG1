# DP05 — California Housing Price Prediction

California Housing 데이터를 이용하여 주택 가격 예측 모델을 만들고,
FastAPI 백엔드와 Streamlit 프론트엔드를 연결하여
사용자가 입력한 주택 정보를 기반으로 가격을 예측하는 시스템을 구현하였다.

---

## 2. 모델 준비: 학습 → 전처리 파이프라인 → 저장

### ✅ 체크포인트

#### 정규화에서 학습 데이터의 통계를 테스트 데이터에도 사용하는 이유는 무엇입니까?

정규화에 사용하는 평균(mean)과 표준편차(std)는 반드시 학습 데이터에서 계산한 값을 사용해야 한다.

테스트 데이터에서 별도로 평균과 표준편차를 계산하면 테스트 데이터의 정보가 모델 처리 과정에 반영되는 데이터 누수(Data Leakage)가 발생할 수 있다.

따라서 학습 데이터에서 계산한 mean과 std를 테스트 데이터와 실제 추론 데이터에도 동일하게 적용해야 모델이 학습할 때와 동일한 기준으로 데이터를 처리할 수 있다.

---

#### 모델 가중치 외에 **함께 저장해야 하는 것**은 무엇이고, 왜 필요합니까?

모델 가중치와 함께 다음과 같은 전처리 정보를 저장해야 한다.

- 피처 이름(feature names)
- 피처 순서
- 정규화에 사용한 평균(mean)
- 정규화에 사용한 표준편차(std)
- 모델 구조 및 설정 정보

실제 서비스에서 새로운 데이터가 들어왔을 때 학습 당시와 동일한 전처리를 적용해야 하기 때문이다.

특히 mean과 std가 달라지거나 피처의 순서가 변경되면 같은 모델을 사용하더라도 잘못된 예측 결과가 나올 수 있다.

---

#### `HousingPredictor.predict()`에서 피처를 `self.feature_names` 순서로 배열하는 이유는?

머신러닝 모델은 학습할 때 사용한 피처의 순서를 기준으로 입력값을 처리한다.

예를 들어 학습 순서가 다음과 같다면,

`MedInc → HouseAge → AveRooms → AveBedrms → Population → AveOccup → Latitude → Longitude`

실제 예측에서도 반드시 같은 순서로 입력해야 한다.

`self.feature_names`를 사용하면 API에서 전달된 딕셔너리의 데이터를 학습 당시의 피처 순서로 정확하게 배열할 수 있다.

---

## 3. FastAPI 백엔드: 추론 엔드포인트 + Pydantic 스키마

### ✅ 체크포인트

#### `HousingRequest`에서 `Latitude`에 `ge=32, le=42` 제한을 넣은 이유는?

`ge=32`는 32 이상, `le=42`는 42 이하라는 의미이다.

California Housing 데이터가 다루는 지역의 위도 범위에 맞지 않는 비정상적인 값이 API에 입력되는 것을 방지하기 위한 입력 검증 조건이다.

Pydantic이 API 요청을 받을 때 해당 범위를 확인하여 잘못된 입력값을 모델에 전달하기 전에 차단할 수 있다.

---

#### `request.model_dump()`는 어떤 역할을 합니까?

`request`는 Pydantic 모델 객체이다.

`request.model_dump()`는 Pydantic 객체에 들어 있는 데이터를 일반 Python 딕셔너리(dict) 형태로 변환한다.

예:

`HousingRequest 객체 → Python dict`

변환된 딕셔너리는 이후 모델의 `predict()` 함수에 전달하여 추론에 사용할 수 있다.

---

#### `run_in_executor`를 사용하지 않으면 어떤 문제가 발생할 수 있습니까?

모델의 `predict()`와 같은 추론 작업은 CPU 또는 GPU를 사용하는 동기 작업일 수 있다.

이 작업을 `async def` 내부에서 직접 실행하면 추론이 끝날 때까지 이벤트 루프가 막힐 수 있다.

그 결과 동시에 여러 사용자의 요청이 들어왔을 때 다른 요청까지 기다려야 하는 문제가 발생할 수 있다.

`run_in_executor`를 사용하면 모델 추론을 별도의 스레드에서 실행하여 FastAPI의 이벤트 루프가 다른 요청을 처리할 수 있도록 한다.

---

## 4. Streamlit 프론트엔드: 입력 폼 → API 호출 → 결과 시각화

### ✅ 체크포인트

#### MNIST 대시보드(Day 4)와 비교했을 때 입력 방식이 어떻게 다릅니까?

MNIST 프로젝트에서는 이미지가 모델의 주요 입력 데이터였다.

이번 California Housing 프로젝트에서는 이미지 대신 사용자가 여러 개의 숫자형 주택 정보를 입력한다.

예를 들어 다음과 같은 값을 입력한다.

- MedInc
- HouseAge
- AveRooms
- AveBedrms
- Population
- AveOccup
- Latitude
- Longitude

즉,

`MNIST : 이미지 입력 → 숫자 분류`

`California Housing : 숫자형 피처 입력 → 주택 가격 예측`

이라는 차이가 있다.

---

#### `st.number_input()`에서 `min_value`, `max_value`를 설정하는 이유는?

사용자가 모델이 예상하지 못한 비정상적인 값을 입력하는 것을 방지하기 위해서이다.

예를 들어 위도와 경도 등에 현실적인 입력 범위를 설정하면 Streamlit 화면에서부터 잘못된 입력값을 제한할 수 있다.

따라서 프론트엔드 단계에서 1차적인 입력값 검증을 수행하는 역할을 한다.

---

#### `request_data` 딕셔너리의 키 이름이 `HousingRequest` 스키마의 필드 이름과 정확히 일치해야 하는 이유는?

Streamlit은 `request_data`를 JSON 형태로 FastAPI의 `/predict` API에 전송한다.

FastAPI에서는 이 JSON 데이터를 `HousingRequest` Pydantic 스키마를 이용하여 검증한다.

따라서 다음과 같이 양쪽의 이름이 일치해야 한다.

`Streamlit request_data → JSON → FastAPI HousingRequest`

예를 들어 FastAPI가 `MedInc`를 요구하는데 Streamlit에서 `med_inc`라는 다른 이름을 보내면 필요한 필드를 찾지 못하여 입력 검증 오류가 발생할 수 있다.

---

## ✅ Day 5 최종 체크포인트

### Q1. 전처리 파라미터(mean, std)를 모델과 함께 저장해야 하는 이유는?

모델 학습 시 사용한 것과 동일한 기준으로 새로운 데이터를 정규화하기 위해서이다.

학습 때와 실제 추론 때 서로 다른 mean과 std를 사용하면 입력 데이터의 스케일이 달라져 모델의 예측 결과가 잘못될 수 있다.

따라서 모델 가중치뿐만 아니라 학습 데이터에서 계산한 mean과 std도 함께 저장해야 한다.

---

### Q2. HousingRequest에서 Latitude에 ge=32, le=42를 넣은 이유는?

California Housing 데이터가 사용하는 지역 범위를 벗어난 비정상적인 위도 입력을 방지하기 위해서이다.

Pydantic을 이용하여 API 단계에서 입력값의 범위를 검증함으로써 잘못된 데이터가 모델까지 전달되는 것을 막을 수 있다.

---

### Q3. Streamlit의 입력값 이름이 Pydantic 스키마의 필드 이름과 일치해야 하는 이유는?

Streamlit에서 만든 입력 데이터는 JSON으로 FastAPI에 전달되고,
FastAPI의 Pydantic 스키마가 해당 JSON의 필드 이름을 기준으로 데이터를 검증한다.

따라서 두 이름이 일치하지 않으면 필요한 필드가 없다고 판단하여 요청 검증 오류가 발생할 수 있다.

---

### Q4. 이 프로젝트에서 run_in_executor를 제거하면 어떤 문제가 생길 수 있습니까?

모델 추론 작업이 FastAPI의 이벤트 루프를 직접 점유할 수 있다.

추론 시간이 길어지면 그동안 다른 요청을 처리하지 못해 서버의 응답성이 떨어질 수 있다.

`run_in_executor`를 사용하면 동기적인 모델 추론을 별도의 스레드에서 실행하여 이벤트 루프가 다른 요청을 계속 처리할 수 있도록 한다.

---

### Q5. MNIST 프로젝트(Day 1~4)와 오늘 프로젝트의 가장 큰 차이는 무엇입니까?

가장 큰 차이는 모델에 전달하는 데이터와 예측 목적이다.

MNIST 프로젝트는 이미지 데이터를 입력받아 숫자 클래스를 분류하는 **분류(Classification)** 문제이다.

반면 California Housing 프로젝트는 여러 개의 숫자형 주택 정보를 입력받아 연속적인 주택 가격을 예측하는 **회귀(Regression)** 문제이다.

즉,

`MNIST → 이미지 → 분류`

`California Housing → 숫자형 피처 → 가격 회귀 예측`

이라는 차이가 있다.

---

## 실행 결과

### 1. 정상 가격 예측

FastAPI 서버가 정상적으로 연결된 상태에서 기본 입력값으로 가격 예측을 실행하였다.

![정상 가격 예측](./images/1_normal_prediction.png)

---

### 2. MedInc 8.0 입력 테스트

중위 소득(MedInc)을 8.0으로 변경한 후 모델의 예측 결과를 확인하였다.

![MedInc 8.0 예측](./images/2_medinc_8_prediction.png)

---

### 3. 서버 연결 오류 테스트

FastAPI 서버를 종료한 후 Streamlit이 서버 연결 실패 상태를 정상적으로 표시하는지 확인하였다.

![서버 연결 오류](./images/3_server_connection_error.png)

---

### 4. Health Check

FastAPI 서버를 다시 실행한 후 `/health` 엔드포인트를 호출하여
서버 상태 코드 `200`과 모델 상태 `healthy`를 확인하였다.

![Health Check](./images/4_health_check.png)

---

## 프로젝트 구조

```text
DP05/
├── DP05.ipynb
├── README.md
└── images/
    ├── 1_normal_prediction.png
    ├── 2_medinc_8_prediction.png
    ├── 3_server_connection_error.png
    └── 4_health_check.png


# DP05 — 모델 배포 통합 프로젝트

## 1. 프로젝트 개요

이번 실습에서는 **California Housing 주택 가격 예측 모델**을 이용하여
사용자가 웹 화면에서 데이터를 입력하고 예측 결과를 확인할 수 있는
모델 배포 시스템을 구현하였다.

전체 구조는 다음과 같다.

```text
사용자
  ↓
Streamlit
  ↓
HTTP 요청
  ↓
FastAPI
  ↓
PyTorch 예측 모델
  ↓
FastAPI 응답
  ↓
Streamlit 예측 결과 표시
```

주요 구성 요소는 다음과 같다.

- **Streamlit** : 사용자가 주택 정보를 입력하는 프론트엔드
- **FastAPI** : 입력 데이터를 받아 모델을 호출하는 백엔드 API
- **PyTorch Model** : California Housing 가격 예측
- **Health Check** : 서버와 모델의 정상 작동 상태 확인


---

# 2. 실행 및 테스트 결과

## 2.1 테스트 1 — 기본값 정상 예측

### 실습 내용

Streamlit 화면에서 기본 주택 정보를 입력한 상태로
**`🚀 가격 예측`** 버튼을 클릭하였다.

FastAPI 서버가 정상적으로 연결되어 있는지 확인하고,
Streamlit에서 입력한 데이터가 FastAPI의 `/predict` API로 전달되어
모델 추론 결과가 다시 Streamlit 화면에 표시되는지 확인하였다.

### 실행 화면

![기본값 정상 예측](./images/1_normal_prediction.png)

### 확인 결과

- 서버 상태 : **🟢 서버 연결됨**
- 중위 소득(MedInc) : `3.50`
- 주택 연식(HouseAge) : `25.00`
- 평균 방 수(AveRooms) : `5.00`
- 평균 침실 수(AveBedrms) : `1.00`
- 예측 결과가 Streamlit 화면에 정상적으로 표시됨

### 확인한 처리 흐름

```text
사용자 입력
   ↓
Streamlit
   ↓
POST /predict
   ↓
FastAPI
   ↓
PyTorch 모델 추론
   ↓
예측 결과 반환
   ↓
Streamlit 화면 출력
```

**결론:** 기본 입력값을 이용한 End-to-End 예측 요청이 정상적으로 처리되는 것을 확인하였다.


---

## 2.2 테스트 2 — 입력값 변경 후 예측

### 실습 내용

모델이 사용자의 입력값 변경을 실제 요청 데이터에 반영하는지 확인하기 위해
중위 소득 **MedInc 값을 3.5에서 8.0으로 변경**하였다.

값을 변경한 후 다시 **`🚀 가격 예측`** 버튼을 클릭하여
새로운 입력 데이터가 FastAPI를 거쳐 모델에 전달되는지 확인하였다.

### 실행 화면

![MedInc 8.0 변경 예측](./images/2_medinc_8_prediction.png)

### 확인 결과

- MedInc : `8.00`
- HouseAge : `25.00`
- AveRooms : `5.00`
- AveBedrms : `1.00`
- Population : `1500.00`
- AveOccup : `3.00`
- Latitude : `37.00`
- Longitude : `-122.00`
- 변경된 입력값을 이용한 새로운 예측 결과가 화면에 표시됨

### 확인한 처리 흐름

```text
MedInc = 8.0 입력
       ↓
Streamlit
       ↓
request_data 생성
       ↓
POST /predict
       ↓
FastAPI
       ↓
모델 추론
       ↓
새로운 예측 결과
```

> **주의:** 특정 입력 변수 하나를 증가시켰다고 해서 예측 가격이 반드시 증가하는 것은 아니다.  
> 모델은 MedInc뿐만 아니라 HouseAge, AveRooms, Population, Latitude,
> Longitude 등 여러 입력 특성을 함께 사용하여 최종 결과를 계산한다.

**결론:** Streamlit에서 변경한 입력값이 API 요청에 반영되고 모델이 새로운 예측 결과를 반환하는 것을 확인하였다.


---

## 2.3 테스트 3 — 서버 연결 실패 및 예외 처리

### 실습 내용

실제 서비스에서는 백엔드 서버가 항상 정상적으로 작동한다고 가정할 수 없다.

따라서 FastAPI 서버를 의도적으로 종료하여
Streamlit이 서버 장애 상황을 어떻게 처리하는지 테스트하였다.

서버 종료에는 다음 명령을 사용하였다.

```python
stop_server(8000)
```

### 실행 화면

![FastAPI 서버 연결 실패](./images/3_server_connection_error.png)

### 확인 결과

FastAPI 서버 종료 후 Streamlit에서 다음 상태가 표시되는 것을 확인하였다.

```text
🔴 서버 연결 실패
서버에 연결할 수 없습니다.
FastAPI 서버를 실행하세요.
```

### 확인한 처리 흐름

```text
Streamlit
   ↓
FastAPI 요청
   ↓
서버 중단
   ↓
ConnectionError 발생
   ↓
예외 처리
   ↓
사용자에게 서버 연결 실패 표시
```

이를 통해 서버 장애가 발생하더라도 프로그램이 단순히 중단되는 것이 아니라,
사용자에게 서버 연결 문제를 알려주는 예외 처리 기능이 동작하는 것을 확인하였다.

**결론:** FastAPI 서버 장애 상황에 대한 기본적인 연결 오류 처리가 정상적으로 동작하였다.


---

## 2.4 테스트 4 — 서버 복구 및 Health Check

### 실습 내용

테스트 3에서 종료한 FastAPI 서버를 다시 실행하였다.

```python
serve_in_thread("app.housing_api:app", port=8000)
```

서버 재실행 후 `/health` API를 호출하여
FastAPI 서버와 모델이 정상 상태인지 확인하였다.

```python
import requests

resp = requests.get("http://localhost:8000/health")

print("[테스트 4] 헬스체크")
print("상태 코드:", resp.status_code)
print("응답:", resp.json())
```

### 실행 화면

![FastAPI Health Check](./images/4_health_check.png)

### 실행 결과

```text
[테스트 4] 헬스체크
상태 코드: 200
응답: {'status': 'healthy', 'model': 'California Housing'}
```

### 확인 결과

- FastAPI 서버 재실행 성공
- `/health` API 호출 성공
- HTTP 상태 코드 : **200**
- 서버 상태 : **healthy**
- 모델 : **California Housing**

### 확인한 처리 흐름

```text
FastAPI 서버 재실행
        ↓
GET /health
        ↓
FastAPI
        ↓
서버 및 모델 상태 확인
        ↓
HTTP 200
        ↓
status = healthy
```

**결론:** 서버 장애 발생 후 FastAPI 서버를 다시 실행하고,
Health Check를 통해 서비스가 정상 상태로 복구된 것을 확인하였다.


---

# 3. 테스트 결과 종합

| 번호 | 테스트 | 주요 확인 내용 | 결과 |
|---|---|---|---|
| 1 | 기본값 정상 예측 | Streamlit → FastAPI → Model → 결과 출력 | 정상 |
| 2 | 입력값 변경 | MedInc 변경값이 모델 요청에 반영되는지 확인 | 정상 |
| 3 | 서버 장애 | FastAPI 종료 시 연결 실패 및 예외 처리 확인 | 정상 |
| 4 | 서버 복구 | `/health` HTTP 200 및 `healthy` 확인 | 정상 |

---

# 4. 최종 시스템 구조

```text
[사용자]
   │
   │ 주택 정보 입력
   ▼
[Streamlit]
   │
   │ POST /predict
   ▼
[FastAPI]
   │
   │ 입력 데이터 검증
   ▼
[Pydantic]
   │
   ▼
[PyTorch Model]
   │
   │ 가격 예측
   ▼
[FastAPI Response]
   │
   ▼
[Streamlit]
   │
   ▼
[예측 결과 표시]
```

---

# 5. 최종 결과

이번 실습에서는 단순히 머신러닝 모델을 실행하는 것을 넘어,

**Streamlit → FastAPI → PyTorch Model**

로 연결되는 모델 서비스 구조를 구현하였다.

또한 다음 네 가지 상황을 직접 테스트하였다.

1. 정상적인 모델 예측
2. 사용자 입력값 변경 후 재예측
3. FastAPI 서버 장애 및 예외 처리
4. 서버 복구 및 Health Check

이를 통해 **사용자 입력 → API 요청 → 모델 추론 → 결과 반환 → 장애 처리 → 서버 상태 확인**까지 이어지는 모델 배포의 기본 End-to-End 흐름을 확인하였다.

---

# 6. 폴더 구조

```text
DP05/
│
├── README.md
│
└── images/
    ├── 1_normal_prediction.png
    ├── 2_medinc_8_prediction.png
    ├── 3_server_connection_error.png
    └── 4_health_check.png
```