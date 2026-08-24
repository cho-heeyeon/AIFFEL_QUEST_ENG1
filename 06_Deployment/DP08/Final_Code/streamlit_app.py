import streamlit as st
import requests


# ==========================================
# 1. 기본 설정
# ==========================================

st.set_page_config(
    page_title="절삭가공 설비 이상 예측",
    page_icon="⚙️",
    layout="centered"
)


# ==========================================
# 2. FastAPI 설정
# ==========================================

API_URL = "http://127.0.0.1:8000/predict"

# auth.py에 설정한 API Key와 동일해야 함
API_KEY = "equipment-secret-key"


# ==========================================
# 3. 화면 제목
# ==========================================

st.title("⚙️ 절삭가공 설비 이상 예측 서비스")

st.write(
    "진동값, 오류 발생 횟수, 전류값을 입력하면 "
    "LogisticRegression 모델이 설비 상태를 "
    "**정상 / 이상**으로 예측합니다."
)


# ==========================================
# 4. 사용자 입력
# ==========================================

vibration = st.number_input(
    "진동값",
    min_value=0.0,
    value=1.5,
    step=0.1
)

error_count = st.number_input(
    "오류 발생 횟수",
    min_value=0,
    value=1,
    step=1
)

current = st.number_input(
    "전류값 (A)",
    min_value=0.0,
    value=9.0,
    step=0.1
)


# ==========================================
# 5. 설비 상태 예측
# ==========================================

if st.button("설비 상태 예측"):

    # FastAPI로 보낼 데이터
    payload = {
        "vibration": vibration,
        "error_count": error_count,
        "current": current
    }

    # API Key를 Header에 넣음
    headers = {
        "X-API-Key": API_KEY
    }

    try:

        response = requests.post(
            API_URL,
            json=payload,
            headers=headers,
            timeout=10
        )

        # ------------------------------
        # 정상 응답
        # ------------------------------

        if response.status_code == 200:

            data = response.json()

            st.subheader("예측 결과")

            st.write(f"진동값: {data['vibration']}")
            st.write(f"오류 발생 횟수: {data['error_count']}")
            st.write(f"전류값: {data['current']} A")

            if data["result"] == "정상":
                st.success("✅ 설비 상태: 정상")

            else:
                st.error("❌ 설비 상태: 이상")

        # ------------------------------
        # 인증 실패
        # ------------------------------

        elif response.status_code == 401:

            st.error("🔑 API Key 인증에 실패했습니다.")

        # ------------------------------
        # 입력값 검증 실패
        # ------------------------------

        elif response.status_code == 422:

            st.error("⚠️ 입력값이 올바르지 않습니다.")

        # ------------------------------
        # 모델 사용 불가
        # ------------------------------

        elif response.status_code == 503:

            st.error("⚠️ 예측 모델을 사용할 수 없습니다.")

        # ------------------------------
        # 기타 서버 오류
        # ------------------------------

        else:

            st.error(
                f"서버 오류가 발생했습니다. "
                f"HTTP 상태 코드: {response.status_code}"
            )

    # FastAPI 서버 연결 실패
    except requests.exceptions.ConnectionError:

        st.error(
            "❌ FastAPI 서버에 연결할 수 없습니다. "
            "FastAPI 서버가 실행 중인지 확인하세요."
        )

    # 요청 시간 초과
    except requests.exceptions.Timeout:

        st.error("⏱️ 서버 응답 시간이 초과되었습니다.")

    # 기타 오류
    except Exception as e:

        st.error(f"예상하지 못한 오류가 발생했습니다: {e}")


# ==========================================
# 6. 프로젝트 안내
# ==========================================

st.divider()

st.caption(
    "※ 본 서비스는 모델 배포 학습을 위한 교육용 프로젝트이며, "
    "가상 데이터로 학습된 모델을 사용합니다."
)