"""
01. PyTorch 기초
================
Transformer를 구현하기 전에 알아야 할 PyTorch 기본 개념들을 배웁니다.

이 파일에서 배우는 것:
1. 텐서(Tensor) - 다차원 배열, 딥러닝의 기본 데이터 구조
2. 자동 미분(Autograd) - 기울기를 자동으로 계산해주는 기능
3. nn.Module - 신경망을 만드는 기본 클래스
4. 손실함수와 옵티마이저 - 학습에 필요한 도구들
"""

import torch
import torch.nn as nn

print("=" * 60)
print("1. 텐서(Tensor) 기초")
print("=" * 60)

# ============================================================
# 텐서란?
# - NumPy의 ndarray와 비슷한 다차원 배열
# - GPU에서 연산 가능 (빠른 병렬 처리)
# - 자동 미분 지원 (신경망 학습에 필수)
# ============================================================

# 텐서 생성 방법 1: 직접 값을 지정
# torch.tensor()에 Python 리스트를 넣으면 텐서가 됩니다
scalar = torch.tensor(3.14)           # 스칼라 (0차원) - 숫자 하나
vector = torch.tensor([1, 2, 3])      # 벡터 (1차원) - 숫자들의 나열
matrix = torch.tensor([[1, 2],
                       [3, 4]])       # 행렬 (2차원) - 숫자들의 표

print(f"스칼라: {scalar}, 차원: {scalar.dim()}, 형태: {scalar.shape}")
print(f"벡터: {vector}, 차원: {vector.dim()}, 형태: {vector.shape}")
print(f"행렬:\n{matrix}, 차원: {matrix.dim()}, 형태: {matrix.shape}")

# 텐서 생성 방법 2: 특정 패턴으로 생성
zeros = torch.zeros(2, 3)    # 0으로 채운 2x3 행렬
ones = torch.ones(2, 3)      # 1로 채운 2x3 행렬
rand = torch.rand(2, 3)      # 0~1 사이 랜덤값으로 채운 2x3 행렬
randn = torch.randn(2, 3)    # 평균 0, 표준편차 1인 정규분포에서 샘플링

print(f"\n영행렬:\n{zeros}")
print(f"랜덤 행렬 (균등분포 0~1):\n{rand}")
print(f"랜덤 행렬 (정규분포):\n{randn}")

# ============================================================
# 텐서 연산
# - 대부분의 수학 연산이 요소별(element-wise)로 적용됩니다
# ============================================================

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"\na = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")           # 요소별 덧셈
print(f"a * b = {a * b}")           # 요소별 곱셈 (Hadamard product)
print(f"a @ b = {a @ b}")           # 내적 (dot product): 1*4 + 2*5 + 3*6 = 32

# 행렬 곱셈 (Matrix Multiplication)
# Transformer에서 가장 많이 사용하는 연산!
A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])      # 2x2 행렬
B = torch.tensor([[5.0, 6.0],
                  [7.0, 8.0]])      # 2x2 행렬

print(f"\n행렬 A:\n{A}")
print(f"행렬 B:\n{B}")
print(f"A @ B (행렬 곱셈):\n{A @ B}")  # 또는 torch.matmul(A, B)

# ============================================================
# 텐서의 형태 변환 (Reshape)
# - 신경망에서 데이터 형태를 맞추는 데 자주 사용
# ============================================================

x = torch.arange(12)  # [0, 1, 2, ..., 11] - 12개의 숫자
print(f"\n원본 텐서: {x}, 형태: {x.shape}")

# view(): 텐서의 형태를 변경 (메모리 공유)
x_reshaped = x.view(3, 4)  # 3행 4열로 변경
print(f"3x4로 변경:\n{x_reshaped}")

x_reshaped = x.view(2, 2, 3)  # 2x2x3 (3차원)으로 변경
print(f"2x2x3으로 변경:\n{x_reshaped}")

# -1을 사용하면 자동으로 크기 계산
x_reshaped = x.view(4, -1)  # 4행, 열은 자동 계산 (12/4=3)
print(f"4x?로 변경 (자동 계산):\n{x_reshaped}")
print(f"원본 텐서: {x}")

print("\n" + "=" * 60)
print("2. 자동 미분 (Autograd)")
print("=" * 60)

# ============================================================
# 자동 미분이란?
# - 함수의 미분(기울기)을 자동으로 계산해주는 기능
# - 신경망 학습에서 가중치를 얼마나 조정할지 결정하는 데 사용
#
# 예: y = x^2 의 미분은 dy/dx = 2x
# x=3일 때 기울기는 2*3 = 6
# ============================================================

# requires_grad=True: "이 텐서에 대한 기울기를 계산해줘"
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2  # y = x^2

print(f"x = {x}")
print(f"y = x^2 = {y}")

# backward(): 역전파 수행 - 기울기 계산
y.backward()

# x.grad: x에 대한 y의 기울기 (dy/dx)
print(f"dy/dx = 2x = 2*3 = {x.grad}")

# 더 복잡한 예제
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2          # [1, 4, 9]
z = y.sum()         # 1 + 4 + 9 = 14

z.backward()        # 역전파

print(f"\nx = {x.data}")
print(f"y = x^2 = {(x**2).data}")
print(f"z = sum(y) = {z.data}")
print(f"dz/dx = 2x = {x.grad}")  # [2, 4, 6]

print("\n" + "=" * 60)
print("3. nn.Module로 신경망 만들기")
print("=" * 60)

# ============================================================
# nn.Module이란?
# - PyTorch에서 신경망을 만드는 기본 클래스
# - 모든 신경망 레이어와 모델은 이 클래스를 상속받아 만듦
# ============================================================


class SimpleNetwork(nn.Module):
    """
    간단한 신경망 예제

    구조: 입력(4) -> 은닉층(8) -> 출력(2)

    이 신경망은:
    1. 4개의 숫자를 입력받아
    2. 8개의 뉴런으로 이루어진 은닉층을 거쳐
    3. 2개의 숫자를 출력합니다
    """

    def __init__(self):
        # 부모 클래스(nn.Module)의 초기화 메서드 호출 - 필수!
        super().__init__()

        # nn.Linear: 선형 변환 레이어 (y = Wx + b)
        # Linear(입력 크기, 출력 크기)
        self.layer1 = nn.Linear(4, 8)   # 4개 입력 -> 8개 출력
        self.layer2 = nn.Linear(8, 2)   # 8개 입력 -> 2개 출력

        # nn.ReLU: 활성화 함수 (음수를 0으로, 양수는 그대로)
        # 활성화 함수가 없으면 아무리 층을 쌓아도 결국 선형 변환
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        순전파(Forward Pass): 입력 -> 출력 계산

        Args:
            x: 입력 텐서, 형태 (배치크기, 4)

        Returns:
            출력 텐서, 형태 (배치크기, 2)
        """
        # 1단계: 첫 번째 선형 변환
        x = self.layer1(x)
        # 2단계: ReLU 활성화 함수 적용
        x = self.relu(x)
        # 3단계: 두 번째 선형 변환
        x = self.layer2(x)
        return x


# 모델 생성
model = SimpleNetwork()
print(f"모델 구조:\n{model}")

# 모델의 파라미터(가중치) 확인
print(f"\n모델 파라미터:")
for name, param in model.named_parameters():
    print(f"  {name}: 형태 {param.shape}")

# 순전파 테스트
# 배치 크기 3, 입력 크기 4인 랜덤 데이터
sample_input = torch.randn(3, 4)
print(f"\n입력 형태: {sample_input.shape}")

output = model(sample_input)  # forward() 메서드가 자동 호출됨
print(f"출력 형태: {output.shape}")
print(f"출력 값:\n{output}")

print("\n" + "=" * 60)
print("4. 손실함수와 옵티마이저")
print("=" * 60)

# ============================================================
# 신경망 학습의 기본 과정:
# 1. 순전파: 입력 -> 예측값 계산
# 2. 손실 계산: 예측값과 정답의 차이 계산
# 3. 역전파: 손실에 대한 각 파라미터의 기울기 계산
# 4. 파라미터 업데이트: 기울기 방향으로 파라미터 조정
# ============================================================

# 학습용 가짜 데이터 생성
X = torch.randn(10, 4)   # 10개 샘플, 각 샘플은 4개 특성
y = torch.randint(0, 2, (10,))  # 10개의 정답 레이블 (0 또는 1)

print(f"입력 데이터 형태: {X.shape}")
print(f"정답 레이블: {y}")

# 손실함수 정의
# CrossEntropyLoss: 분류 문제에서 가장 많이 사용
# 예측 확률분포와 실제 정답 사이의 차이를 측정
criterion = nn.CrossEntropyLoss()

# 옵티마이저 정의
# SGD (Stochastic Gradient Descent): 가장 기본적인 최적화 알고리즘
# lr (learning rate): 학습률 - 한 번에 얼마나 크게 파라미터를 조정할지
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 학습 루프 (5 에폭)
print("\n학습 시작:")
for epoch in range(5):
    # 1. 순전파
    predictions = model(X)  # 예측값 계산

    # 2. 손실 계산
    loss = criterion(predictions, y)

    # 3. 역전파 준비: 기존 기울기 초기화
    # (안 하면 기울기가 누적됨)
    optimizer.zero_grad()

    # 4. 역전파: 기울기 계산
    loss.backward()

    # 5. 파라미터 업데이트
    optimizer.step()

    print(f"  에폭 {epoch + 1}: 손실 = {loss.item():.4f}")

print("\n" + "=" * 60)
print("5. 배치 처리와 차원 이해 (Transformer에서 중요!)")
print("=" * 60)

# ============================================================
# Transformer에서 자주 사용하는 텐서 형태:
# (batch_size, sequence_length, embedding_dim)
#
# - batch_size: 한 번에 처리하는 데이터 개수
# - sequence_length: 문장의 길이 (토큰 개수)
# - embedding_dim: 각 토큰을 표현하는 벡터의 크기
# ============================================================

batch_size = 2       # 2개의 문장을 동시에 처리
seq_length = 5       # 각 문장은 5개의 토큰으로 구성
embed_dim = 8        # 각 토큰은 8차원 벡터로 표현

# 예시 데이터: (2, 5, 8) 형태
# - 2개의 문장
# - 각 문장에 5개의 토큰
# - 각 토큰은 8차원 벡터
data = torch.randn(batch_size, seq_length, embed_dim)

print(f"데이터 형태: {data.shape}")
print(f"  - 배치 크기: {batch_size}")
print(f"  - 시퀀스 길이: {seq_length}")
print(f"  - 임베딩 차원: {embed_dim}")

# 첫 번째 문장의 첫 번째 토큰
print(f"\n첫 번째 문장의 첫 번째 토큰 벡터:")
print(f"  {data[0, 0, :]}")

# 차원 접근 예시
print(f"\n첫 번째 문장 전체: 형태 {data[0].shape}")  # (5, 8)
print(f"모든 문장의 첫 번째 토큰: 형태 {data[:, 0, :].shape}")  # (2, 8)

print("\n" + "=" * 60)
print("핵심 요약")
print("=" * 60)
print("""
1. 텐서(Tensor): 다차원 배열, GPU 연산과 자동 미분 지원
2. 자동 미분: backward()로 기울기 자동 계산
3. nn.Module: 신경망의 기본 클래스, forward() 메서드 구현 필요
4. 학습 루프: 순전파 → 손실계산 → 역전파 → 파라미터 업데이트
5. 텐서 형태: Transformer에서는 (batch, seq_len, embed_dim)이 기본

다음 단계에서는 Transformer의 핵심인 Attention 메커니즘을 배웁니다!
""")
