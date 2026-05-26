# Q 1 : 클로저를 활용한 문제

def find_min_max(numbers):
    # min_value와 max_value 변수를 초기화
    # min_value는 양의 무한대(float('inf'))로 초기화하여 어떤 숫자보다도 큰 값으로 설정
    min_value = float('inf')
    # max_value는 음의 무한대(float('-inf'))로 초기화하여 어떤 숫자보다도 작은 값으로 설정
    max_value = float('-inf')

    def update_min_max(num):
        # 외부함수의 변수인 min_value, max_value 참조
        nonlocal min_value
        nonlocal max_value
        # 만약 num 값이 min_value보다 작다면 min_value를 num 값으로 변경
        if num < min_value :
            min_value = num
            
        # 만약 num 값이 max_value보다 크다면 max_value를 num 값으로 변경
        if num > max_value :
            max_value = num
   
    # numbers 리스트의 모든 값을 순환하며 최댓값과 최솟값 업데이트
    for num in numbers:
        update_min_max(num)

    # 최솟값을 반환하는 내부함수
    def get_min():

        return min_value

    # 최댓값을 반환하는 내부함수
    def get_max():

        return max_value
    # 외부함수는 내부함수(get_min()과 get_max())를 반환
    return get_min,get_max
   
numbers = [10, 5, 8, 12, 3, 7]
find_min, find_max = find_min_max(numbers)

print("최솟값:", find_min())  # 3
print("최댓값:", find_max())  # 12

```
    최솟값: 3
    최댓값: 12




```python
# Q2 : 클로저를 활용한 문제

def counter(fn):
    count = 0
    def counter1():
        nonlocal count
        count += 1
        fn()
        print(fn.__name__, "실행 횟수 :",count)
    return counter1



@counter
def say_hello():
    print("Hello AIFFEL")

for i in range(5):
    say_hello()
    
```
    Hello AIFFEL
    say_hello 실행 횟수 : 1
    Hello AIFFEL
    say_hello 실행 횟수 : 2
    Hello AIFFEL
    say_hello 실행 횟수 : 3
    Hello AIFFEL
    say_hello 실행 횟수 : 4
    Hello AIFFEL
    say_hello 실행 횟수 : 5
