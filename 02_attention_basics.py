"""
02. Attention 메커니즘 기초
==========================
Transformer의 핵심인 Attention 메커니즘을 이해하고 구현합니다.

이 파일에서 배우는 것:
1. Attention이란 무엇인가? (직관적 이해)
2. Query, Key, Value의 개념
3. Scaled Dot-Product Attention 구현
4. Attention 가중치 시각화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

print("=" * 60)
print("1. Attention이란? (직관적 이해)")
print("=" * 60)

# ============================================================
# Attention을 한 문장으로:
# "입력의 어떤 부분에 얼마나 집중할지 결정하는 메커니즘"
#
# 예시: "The cat sat on the mat because it was tired"
# "it"이 무엇을 가리키는지 알려면?
# → "cat"에 높은 attention, "mat"에 낮은 attention
#
# 기존 RNN의 문제:
# - 순서대로 처리 → 먼 거리의 단어 관계 파악 어려움
# - "it"과 "cat" 사이에 여러 단어가 있으면 연결이 약해짐
#
# Attention의 해결:
# - 모든 단어가 모든 단어를 직접 참조 가능
# - 거리에 상관없이 관련 있는 단어에 집중
# ============================================================

print("""
Attention의 핵심 아이디어:
- 입력 시퀀스의 각 위치가 다른 모든 위치를 "참조"할 수 있음
- 참조할 때 "얼마나 중요한지" 가중치를 계산
- 중요한 위치의 정보를 더 많이 가져옴

예: "The cat sat on the mat because it was tired"
    "it"이 참조할 때 → "cat"에 높은 가중치 (관련 있음)
                     → "mat"에 낮은 가중치 (관련 적음)
""")

print("\n" + "=" * 60)
print("2. Query, Key, Value 개념")
print("=" * 60)

# ============================================================
# Query, Key, Value는 데이터베이스 검색에서 유래한 개념입니다.
#
# 비유: 도서관에서 책 찾기
# - Query (질문): "머신러닝 관련 책 찾아줘"
# - Key (색인): 각 책의 주제/카테고리 (머신러닝, 요리, 소설...)
# - Value (내용): 실제 책의 내용
#
# 검색 과정:
# 1. Query와 각 Key를 비교 → 유사도 점수 계산
# 2. 유사도가 높은 책에 더 집중 (가중치)
# 3. 가중치를 적용하여 Value들을 조합 → 최종 결과
#
# Transformer에서:
# - Query: "내가 알고 싶은 것" (현재 처리 중인 단어)
# - Key: "나를 설명하는 것" (모든 단어의 특성)
# - Value: "내가 제공하는 정보" (모든 단어의 실제 내용)
# ============================================================

print("""
Query, Key, Value 비유 (도서관 검색):

┌─────────────────────────────────────────────────────────┐
│ Query: "머신러닝 책 찾아줘"                              │
│                                                         │
│ 책장 (Key → Value):                                     │
│   [머신러닝] → 딥러닝 교과서     → 유사도 높음 (0.9)    │
│   [요리]     → 한식 레시피       → 유사도 낮음 (0.1)    │
│   [AI]       → 인공지능 입문     → 유사도 중간 (0.7)    │
│                                                         │
│ 결과: 0.9×딥러닝교과서 + 0.1×한식레시피 + 0.7×AI입문    │
│       (가중 합으로 결과 생성)                           │
└─────────────────────────────────────────────────────────┘

Transformer에서는:
- 모든 단어가 Query도 되고, Key도 되고, Value도 됩니다
- 각 단어가 "다른 단어들 중 어디에 집중할까?" 계산
""")

print("\n" + "=" * 60)
print("3. Scaled Dot-Product Attention 구현")
print("=" * 60)

# ============================================================
# Attention 계산 공식:
#
# Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
#
# 단계별 설명:
# 1. Q × K^T: Query와 Key의 내적 → 유사도 점수
# 2. / √d_k: 스케일링 (값이 너무 커지는 것 방지)
# 3. softmax: 점수를 확률로 변환 (합이 1)
# 4. × V: 가중치를 적용하여 Value 조합
# ============================================================


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Scaled Dot-Product Attention 구현

    Args:
        query: 질의 텐서, 형태 (batch, seq_len, d_k)
        key: 키 텐서, 형태 (batch, seq_len, d_k)
        value: 값 텐서, 형태 (batch, seq_len, d_v)
        mask: 마스크 텐서 (선택사항)

    Returns:
        output: attention 적용 결과, 형태 (batch, seq_len, d_v)
        attention_weights: attention 가중치, 형태 (batch, seq_len, seq_len)
    """
    # d_k: Key/Query의 차원 (마지막 차원)
    d_k = query.size(-1)

    # 1단계: Q × K^T (Query와 Key의 내적)
    # query: (batch, seq_len, d_k)
    # key.transpose(-2, -1): (batch, d_k, seq_len)
    #   → transpose(-2, -1): 마지막 두 차원(seq_len, d_k)만 교환
    #   → 3D 텐서에서 각 batch를 독립적인 2D 행렬로 보고 전치
    #   → batch 차원은 그대로 유지됨
    # 결과: (batch, seq_len, seq_len)
    scores = torch.matmul(query, key.transpose(-2, -1))

    # 2단계: √d_k로 나누기 (스케일링)
    # 왜? 내적 값이 d_k에 비례해서 커지면 softmax가 극단적이 됨
    # 예: [100, 1, 1] → softmax → [0.99, 0.005, 0.005] (거의 one-hot)
    # 스케일링하면 값이 적당해져서 부드러운 분포 유지
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    # 마스크 적용 (선택사항 - 나중에 디코더에서 사용)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 3단계: Softmax로 확률 변환
    # 각 Query에 대해, 모든 Key에 대한 가중치의 합이 1이 됨
    attention_weights = F.softmax(scores, dim=-1)

    # 4단계: 가중치를 Value에 적용
    # attention_weights: (batch, seq_len, seq_len)
    # value: (batch, seq_len, d_v)
    # 결과: (batch, seq_len, d_v)
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


# 테스트해보기
print("\n간단한 예제로 Attention 이해하기:")
print("-" * 40)

# 3개의 단어로 이루어진 문장 (배치 크기 1)
# 각 단어는 4차원 벡터로 표현
batch_size = 1
seq_len = 3  # 3개 단어: "I", "love", "AI"
d_model = 4  # 각 단어의 임베딩 차원

# 임베딩 벡터 (실제로는 학습되지만, 여기선 예시로 직접 설정)
# 비슷한 의미의 단어는 비슷한 벡터를 가진다고 가정
embeddings = torch.tensor([
    [[1.0, 0.0, 0.0, 0.0],   # "I" - 주어
     [0.0, 1.0, 0.5, 0.0],   # "love" - 동사
     [0.0, 0.5, 1.0, 0.0]]   # "AI" - 목적어 (love와 약간 유사)
])

print(f"입력 임베딩 (3개 단어, 4차원):")
print(f"  'I':    {embeddings[0, 0].tolist()}")
print(f"  'love': {embeddings[0, 1].tolist()}")
print(f"  'AI':   {embeddings[0, 2].tolist()}")

# 단순화를 위해 Q, K, V를 동일하게 설정 (Self-Attention)
# 실제로는 각각 다른 선형 변환을 거침
Q = K = V = embeddings

# Attention 계산
output, attention_weights = scaled_dot_product_attention(Q, K, V)

print(f"\nAttention 가중치 (각 단어가 다른 단어에 얼마나 집중하는지):")
print(f"         →  'I'    'love'  'AI'")
for i, word in enumerate(["'I'   ", "'love'", "'AI'  "]):
    weights = attention_weights[0, i].tolist()
    print(f"  {word}  [{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}]")

print(f"\n해석:")
print(f"  - 'I'는 주로 자기 자신에게 집중 (독립적인 주어)")
print(f"  - 'love'는 자신과 'AI'에 집중 (동사-목적어 관계)")
print(f"  - 'AI'는 자신과 'love'에 집중 (목적어-동사 관계)")

print("\n" + "=" * 60)
print("4. 실제 Transformer에서의 Self-Attention")
print("=" * 60)

# ============================================================
# Self-Attention:
# - 같은 시퀀스 내에서 각 위치가 다른 모든 위치를 참조
# - Q, K, V가 모두 같은 입력에서 나옴 (다른 선형 변환 적용)
#
# 실제 구현에서는:
# Q = input × W_Q (입력에 Query 가중치 행렬 적용)
# K = input × W_K (입력에 Key 가중치 행렬 적용)
# V = input × W_V (입력에 Value 가중치 행렬 적용)
# ============================================================


class SelfAttention(nn.Module):
    """
    Self-Attention 레이어

    입력 시퀀스의 각 위치가 다른 모든 위치를 참조하여
    문맥을 반영한 새로운 표현을 생성합니다.
    """

    def __init__(self, d_model):
        """
        Args:
            d_model: 입력/출력 차원 (임베딩 크기)
        """
        super().__init__()
        self.d_model = d_model

        # Q, K, V를 만들기 위한 선형 변환
        # 각각 다른 "관점"에서 입력을 바라보게 함
        self.W_Q = nn.Linear(d_model, d_model)  # Query 변환
        self.W_K = nn.Linear(d_model, d_model)  # Key 변환
        self.W_V = nn.Linear(d_model, d_model)  # Value 변환

    def forward(self, x, mask=None):
        """
        Args:
            x: 입력 텐서, 형태 (batch, seq_len, d_model)
            mask: 마스크 텐서 (선택사항)

        Returns:
            output: 형태 (batch, seq_len, d_model)
            attention_weights: 형태 (batch, seq_len, seq_len)
        """
        # 입력을 Q, K, V로 변환
        Q = self.W_Q(x)  # (batch, seq_len, d_model)
        K = self.W_K(x)  # (batch, seq_len, d_model)
        V = self.W_V(x)  # (batch, seq_len, d_model)

        # Scaled Dot-Product Attention 적용
        output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        return output, attention_weights


# Self-Attention 테스트
print("\nSelf-Attention 레이어 테스트:")
print("-" * 40)

d_model = 8
seq_len = 4
batch_size = 1

# Self-Attention 레이어 생성
self_attention = SelfAttention(d_model)

# 랜덤 입력 (4개 단어, 8차원 임베딩)
x = torch.randn(batch_size, seq_len, d_model)

print(f"입력 형태: {x.shape}")
print(f"  - 배치 크기: {batch_size}")
print(f"  - 시퀀스 길이: {seq_len}")
print(f"  - 임베딩 차원: {d_model}")

# Self-Attention 적용
output, attention_weights = self_attention(x)

print(f"\n출력 형태: {output.shape}")
print(f"Attention 가중치 형태: {attention_weights.shape}")

print("\n" + "=" * 60)
print("5. Attention 가중치 시각화")
print("=" * 60)


def visualize_attention(attention_weights, tokens, title="Attention Weights"):
    """
    Attention 가중치를 히트맵으로 시각화

    Args:
        attention_weights: (seq_len, seq_len) 형태의 가중치
        tokens: 토큰 리스트
        title: 그래프 제목
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_weights.detach().numpy(), cmap='Blues')
    plt.colorbar(label='Attention Weight')

    # 축 레이블 설정
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.yticks(range(len(tokens)), tokens)
    plt.xlabel('Key (참조되는 단어)')
    plt.ylabel('Query (참조하는 단어)')
    plt.title(title)

    plt.tight_layout()
    plt.savefig('attention_visualization.png', dpi=100)
    plt.close()
    print(f"  시각화 저장됨: attention_visualization.png")


# 예시 문장으로 시각화
print("\n예시 문장의 Attention 시각화:")
print("-" * 40)

# 예시 토큰들
tokens = ["The", "cat", "sat", "on", "mat"]
seq_len = len(tokens)
d_model = 16

# Self-Attention 레이어
self_attention = SelfAttention(d_model)

# 가상의 임베딩 (실제로는 학습된 임베딩 사용)
# 관련 있는 단어들이 비슷한 벡터를 가지도록 설정
embeddings = torch.randn(1, seq_len, d_model)

# Attention 계산
output, attention_weights = self_attention(embeddings)

# 시각화
visualize_attention(attention_weights[0], tokens, "Self-Attention: 'The cat sat on mat'")

print(f"\nAttention 가중치 행렬:")
print(f"(각 행 = Query, 각 열 = Key)")
print(f"         ", end="")
for token in tokens:
    print(f"{token:>6}", end=" ")
print()

for i, token in enumerate(tokens):
    print(f"{token:>6}  ", end="")
    for j in range(len(tokens)):
        print(f"{attention_weights[0, i, j].item():>6.3f}", end=" ")
    print()

print("\n" + "=" * 60)
print("핵심 요약")
print("=" * 60)
print("""
1. Attention: 입력의 어떤 부분에 집중할지 결정하는 메커니즘

2. Query, Key, Value:
   - Query: "내가 찾는 것" (현재 위치)
   - Key: "나를 설명하는 것" (모든 위치의 특성)
   - Value: "내가 제공하는 정보" (모든 위치의 내용)

3. Scaled Dot-Product Attention:
   Attention(Q,K,V) = softmax(QK^T / √d_k) × V
   - QK^T: 유사도 계산
   - √d_k: 스케일링 (안정적인 학습)
   - softmax: 가중치로 변환
   - × V: 가중 합

4. Self-Attention:
   - 같은 시퀀스 내에서 모든 위치가 서로 참조
   - Q, K, V 모두 같은 입력에서 변환

다음 단계에서는 Multi-Head Attention을 배웁니다!
(여러 개의 Attention을 병렬로 수행하여 다양한 관점 학습)
""")
