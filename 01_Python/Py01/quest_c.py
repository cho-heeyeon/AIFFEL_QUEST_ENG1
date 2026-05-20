"""[문제 1] ------ Q1-1 회문 판독기
[문제작성]
문자열을 입력으로 받아 회문이면 True, 아니면 False를 반환
[정답]
문자열을 앞에서부터 읽은 값과 뒤에서부터 읽은 값이 같은지 비교
[실행결과]
예시 입력을 함수에 넣어 True/False 출력

[문제 2] ------ Q1-2 회문 분류기
[문제작성]
단어 리스트를 회문/비회문으로 분류하고 회문 개수, 회문 비율 반환
[정답]
Q1-1 판독 함수를 이용해 단어를 분류 딕셔너리에 누적, 개수/비율 계산
[실행결과]
예시 리스트를 넣어 (분류딕셔너리, 회문개수, 회문비율) 출력

[문제 3] ------ Q2-1 성적 데이터화
[문제작성]
성적 데이터를 딕셔너리로 저장
[정답]
학생 이름 -> 과목별 점수(국어/영어/수학) 딕셔너리 형태로 구성
[실행결과]
scores 출력

[문제 4] ------ Q2-2 성적 분석기
[문제작성]
학생 이름을 받아 과목별 점수와 학급 평균 대비 차이, 전과목 평균 분석
[정답]
학급 평균(과목별/전과목)을 계산한 뒤, 선택 학생 점수와 비교하여 리포트 출력
[실행결과]
예시 학생(설진민) 성적 분석 리포트 출력
"""


def is_palindrome(word):
    s = str(word)
    return s == s[::-1]


def palindrome_classifier(words):
    result = {"회문": [], "회문X": []}

    for w in words:
        if is_palindrome(w):
            result["회문"].append(w)
        else:
            result["회문X"].append(w)

    palindrome_count = len(result["회문"])
    total = len(words)
    palindrome_ratio = palindrome_count / total if total else 0.0

    return result, palindrome_count, palindrome_ratio


scores = {
    "유원영": {"국어": 94, "영어": 91, "수학": 89},
    "설진민": {"국어": 88, "영어": 96, "수학": 90},
    "신윤아": {"국어": 86, "영어": 88, "수학": 77},
    "장윤아": {"국어": 98, "영어": 76, "수학": 92},
}


def _class_subject_averages(score_dict):
    if not score_dict:
        return {}

    subjects = list(next(iter(score_dict.values())).keys())
    avg = {}
    for sub in subjects:
        vals = [v[sub] for v in score_dict.values()]
        avg[sub] = sum(vals) / len(vals)
    return avg


def _student_subject_averages(student_scores):
    if not student_scores:
        return 0.0
    vals = list(student_scores.values())
    return sum(vals) / len(vals)


def score_analysis(name):
    if name not in scores:
        raise KeyError(f"'{name}' 학생 데이터가 없습니다.")

    student = scores[name]
    class_avg_by_subject = _class_subject_averages(scores)

    lines = []
    lines.append(f"[{name} 성적 분석]")

    for subject in ["국어", "영어", "수학"]:
        score = float(student[subject])
        class_avg = float(class_avg_by_subject[subject])
        diff = score - class_avg
        lines.append(
            f"{subject}: {int(score)}점, 학급 평균({class_avg:.2f}점) 대비 {diff:+.2f}"
        )

    lines.append("")

    student_avg = _student_subject_averages(student)
    class_overall_avg = sum(_student_subject_averages(s) for s in scores.values()) / len(scores)
    overall_diff = student_avg - class_overall_avg

    lines.append(
        f"전과목 평균: {student_avg:.2f}점, 학급 평균({class_overall_avg:.2f}점) 대비 {overall_diff:+.2f}"
    )

    return "\n".join(lines)


if __name__ == "__main__":
    print("[문제 1] ------ Q1-1 회문 판독기")
    print("[문제작성]")
    print("문자열을 입력으로 받아 회문이면 True, 아니면 False를 반환하는 함수를 만드세요.")
    print("[정답]")
    print("문자열을 앞에서부터 읽은 값과 뒤에서부터 읽은 값이 같은지 비교합니다.")
    print("[실행결과]")
    target1 = "아만남다다남만아"
    target2 = "아름다운날들부터"
    print(is_palindrome(target1))
    print(is_palindrome(target2))

    print("\n[문제 2] ------ Q1-2 회문 분류기")
    print("[문제작성]")
    print("단어 리스트를 입력으로 받아 회문/비회문으로 분류한 딕셔너리와 회문 개수, 회문 비율을 반환하세요.")
    print("[정답]")
    print("Q1-1 판독 함수를 이용해 단어를 회문/비회문으로 분류하고, 회문 개수와 비율을 계산합니다.")
    print("[실행결과]")
    targets = [
        "기러기",
        "개미",
        "토마토",
        "스윗쏘스",
        "스위스",
        "인도인",
        "한국인",
        "벌벌벌",
        "우영우",
        "우분우",
    ]
    print(palindrome_classifier(targets))

    print("\n[문제 3] ------ Q2-1 성적 데이터화")
    print("[문제작성]")
    print("시험 성적 데이터를 하나의 변수에 딕셔너리 형태로 저장하세요.")
    print("[정답]")
    print("학생 이름을 key로 하고, 과목별 점수를 value(딕셔너리)로 갖는 중첩 딕셔너리로 구성합니다.")
    print("[실행결과]")
    print(scores)

    print("\n[문제 4] ------ Q2-2 성적 분석기")
    print("[문제작성]")
    print("학생 이름을 입력받아 과목별 점수, 학급 평균 대비 차이, 전과목 평균을 분석해 출력하는 함수를 만드세요.")
    print("[정답]")
    print("학급 평균(과목별/전과목)을 구하고, 선택 학생 점수와의 차이를 계산해 리포트 문자열로 반환합니다.")
    print("[실행결과]")
    print(score_analysis("설진민"))
