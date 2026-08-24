from fastapi import Header, HTTPException

# 교육용 프로젝트 API Key
API_KEY = "equipment-secret-key"


def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    """
    요청 Header의 X-API-Key를 확인합니다.
    """

    # API Key가 없는 경우
    if x_api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API Key가 필요합니다."
        )

    # API Key가 잘못된 경우
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="잘못된 API Key입니다."
        )

    return x_api_key