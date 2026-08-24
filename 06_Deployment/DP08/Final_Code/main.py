import asyncio
import joblib

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field

from auth import verify_api_key


# ==========================================
# 1. FastAPI 애플리케이션
# ==========================================

app = FastAPI(
    title="절삭가공 설비 이상 예측 API",
    description="진동값, 오류 발생 횟수, 전류값을 이용하여 설비 상태를 예측합니다.",
    version="1.0"
)


# ==========================================
# 2. 저장된 모델 불러오기
# ==========================================

try:
    model = joblib.load("equipment_model.pkl")
except Exception as e:
    model = None
    print("모델 로드 오류:", e)


# ==========================================
# 3. 입력 데이터 정의
# ==========================================

class EquipmentInput(BaseModel):

    vibration: float = Field(
        ...,
        ge=0,
        description="설비 진동값"
    )

    error_count: int = Field(
        ...,
        ge=0,
        description="오류 발생 횟수"
    )

    current: float = Field(
        ...,
        ge=0,
        description="전류값(A)"
    )


# ==========================================
# 4. 서버 상태 확인 API
# ==========================================

@app.get("/health")
def health():

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="모델을 사용할 수 없습니다."
        )

    return {
        "status": "ok",
        "model": "LogisticRegression"
    }


# ==========================================
# 5. 실제 모델 추론 함수
# ==========================================

def run_prediction(data: EquipmentInput):

    X = [[
        data.vibration,
        data.error_count,
        data.current
    ]]

    prediction = model.predict(X)[0]

    return "정상" if prediction == 1 else "이상"


# ==========================================
# 6. 설비 이상 예측 API
# ==========================================

@app.post("/predict")
async def predict(
    data: EquipmentInput,
    api_key: str = Depends(verify_api_key)
):

    # 모델이 정상적으로 로드되지 않은 경우
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="모델을 사용할 수 없습니다."
        )

    try:
        # 현재 이벤트 루프 가져오기
        loop = asyncio.get_running_loop()

        # 동기식 ML 추론을 별도 스레드에서 실행
        result = await loop.run_in_executor(
            None,
            run_prediction,
            data
        )

        return {
            "vibration": data.vibration,
            "error_count": data.error_count,
            "current": data.current,
            "result": result
        }

    except HTTPException:
        raise

    except Exception as e:
        print("예측 오류:", e)

        raise HTTPException(
            status_code=500,
            detail="모델 예측 중 오류가 발생했습니다."
        )