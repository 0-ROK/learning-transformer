"""
03. Multi-Head Attention
========================
여러 개의 Attention을 병렬로 수행하여 다양한 관점에서 관계를 학습합니다.

이 파일에서 배우는 것:
1. 왜 Multi-Head가 필요한가? (단일 헤드의 한계)
2. Multi-Head Attention의 구조와 수식
3. 차원 이해 (d_model, num_heads, d_k)
4. PyTorch로 직접 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("1. 왜 Multi-Head Attention이 필요한가?")
print("=" * 60)

# ============================================================
# 단일 헤드 Attention의 한계
#
# 예문: "The cat sat on the mat because it was tired"
#
# 이 문장에는 여러 종류의 관계가 있습니다:
# - 대명사 관계: "it" → "cat" (it이 가리키는 것)
# - 문법적 관계: "sat" → "on" → "mat" (동사-전치사-목적어)
# - 의미적 관계: "tired" → "cat" (피곤한 주체)
#
# 문제: 하나의 Attention 헤드는 하나의 "관점"만 잘 학습합니다.
# - W_Q, W_K, W_V가 하나씩만 있으면
# - 모든 관계를 동시에 포착하기 어려움
#
# 해결: 여러 개의 Attention을 병렬로 수행!
# - 각 헤드가 서로 다른 관계 패턴 학습
# - Head 1: 문법적 관계 전문
# - Head 2: 의미적 관계 전문
# - Head 3: 위치적 관계 전문
# - ...
# ============================================================

print("""
단일 헤드의 한계:
┌─────────────────────────────────────────────────────────┐
│ "The cat sat on the mat because it was tired"          │
│                                                         │
│ 포착해야 할 관계들:                                      │
│   1. it → cat (대명사가 가리키는 것)                    │
│   2. sat → on → mat (동사-전치사-목적어)                │
│   3. tired → cat (상태의 주체)                          │
│                                                         │
│ 단일 헤드: 하나의 관계만 잘 포착                         │
│ Multi-Head: 여러 관계를 동시에 포착!                    │
└─────────────────────────────────────────────────────────┘
""")

print("\n" + "=" * 60)
print("2. Multi-Head Attention 구조")
print("=" * 60)

# ============================================================
# Multi-Head Attention 수식:
#
# MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h) × W_O
#
# 여기서 각 헤드는:
# head_i = Attention(Q × W_Q^i, K × W_K^i, V × W_V^i)
#
# 핵심 아이디어:
# 1. 입력을 여러 "헤드"로 나눔
# 2. 각 헤드에서 독립적으로 Attention 수행
# 3. 결과를 합쳐서 (concat) 최종 출력
#
# 차원 설계 (매우 중요!):
# - d_model: 전체 모델 차원 (예: 512)
# - num_heads: 헤드 개수 (예: 8)
# - d_k = d_model / num_heads: 각 헤드의 차원 (예: 64)
#
# 이렇게 하면 총 파라미터 수가 단일 헤드와 동일합니다!
# - 단일 헤드: W_Q, W_K, W_V 각각 (512 × 512)
# - Multi-Head: 8개 헤드 × (512 × 64) = (512 × 512)
# ============================================================

print("""
Multi-Head Attention 구조:

입력: (batch, seq_len, d_model)
      예: (32, 10, 512)

┌─────────────────────────────────────────────────────────┐
│                    Linear (W_Q, W_K, W_V)               │
│                    ↓                                    │
│         ┌─────┬─────┬─────┬─────┬─────┐                │
│         │Head1│Head2│Head3│ ... │Head8│  (8개로 분할)   │
│         └──┬──┴──┬──┴──┬──┴─────┴──┬──┘                │
│            ↓     ↓     ↓           ↓                   │
│         Attention 계산 (각 헤드에서 독립적으로)          │
│            ↓     ↓     ↓           ↓                   │
│         ┌──┴──┬──┴──┬──┴──┬─────┬──┴──┐                │
│         │     │     │     │     │     │  (Concat)      │
│         └─────┴─────┴─────┴─────┴─────┘                │
│                    ↓                                    │
│              Linear (W_O)                               │
└─────────────────────────────────────────────────────────┘

출력: (batch, seq_len, d_model)
      예: (32, 10, 512)
""")

print("\n" + "=" * 60)
print("3. 차원 이해하기")
print("=" * 60)

# ============================================================
# 차원 계산 예시 (d_model=512, num_heads=8)
#
# 1. 입력: (batch, seq_len, 512)
#
# 2. Linear 변환 후: Q, K, V 각각 (batch, seq_len, 512)
#
# 3. 헤드로 분할: (batch, seq_len, 8, 64)
#    → reshape: (batch, 8, seq_len, 64)
#    → 각 헤드가 64차원에서 Attention 수행
#
# 4. Attention 후: 각 헤드 (batch, 8, seq_len, 64)
#
# 5. Concat: (batch, seq_len, 512)
#
# 6. 최종 Linear: (batch, seq_len, 512)
# ============================================================

print("""
차원 흐름 (d_model=512, num_heads=8, d_k=64):

입력: (batch, seq_len, 512)
        ↓
Linear Q,K,V: (batch, seq_len, 512)
        ↓
Split heads: (batch, seq_len, 8, 64)
        ↓
Transpose: (batch, 8, seq_len, 64)
        ↓
Attention: 각 헤드에서 (batch, seq_len, seq_len) 가중치 계산
        ↓
Output: (batch, 8, seq_len, 64)
        ↓
Transpose back: (batch, seq_len, 8, 64)
        ↓
Concat (reshape): (batch, seq_len, 512)
        ↓
Linear W_O: (batch, seq_len, 512)

핵심: d_model = num_heads × d_k
      512 = 8 × 64
""")

print("\n" + "=" * 60)
print("4. Scaled Dot-Product Attention (2단계 복습)")
print("=" * 60)


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Scaled Dot-Product Attention
    (2단계에서 배운 함수와 동일)

    Args:
        query: (batch, num_heads, seq_len, d_k)
        key: (batch, num_heads, seq_len, d_k)
        value: (batch, num_heads, seq_len, d_v)
        mask: 마스크 텐서 (선택사항)

    Returns:
        output: (batch, num_heads, seq_len, d_v)
        attention_weights: (batch, num_heads, seq_len, seq_len)
    """
    d_k = query.size(-1)

    # Q × K^T / √d_k
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    # 마스크 적용 (선택사항)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax
    attention_weights = F.softmax(scores, dim=-1)

    # × V
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


print("2단계에서 배운 Attention 함수를 재사용합니다.")
print("차이점: 이제 num_heads 차원이 추가됩니다!")

print("\n" + "=" * 60)
print("5. Multi-Head Attention 구현")
print("=" * 60)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention 레이어

    여러 개의 Attention 헤드를 병렬로 수행하여
    다양한 관점에서 입력 간의 관계를 학습합니다.
    """

    def __init__(self, d_model, num_heads):
        """
        Args:
            d_model: 모델의 전체 차원 (예: 512)
            num_heads: Attention 헤드 개수 (예: 8)
        """
        super().__init__()

        # d_model이 num_heads로 나누어 떨어져야 함
        assert d_model % num_heads == 0, \
            f"d_model({d_model})은 num_heads({num_heads})로 나누어 떨어져야 합니다"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 각 헤드의 차원

        # Q, K, V를 만들기 위한 선형 변환
        # 전체 차원(d_model)에서 전체 차원(d_model)으로 변환
        # 내부적으로 num_heads개의 (d_model → d_k) 변환이 합쳐진 것
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # 최종 출력을 위한 선형 변환
        self.W_O = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        """
        텐서를 여러 헤드로 분할

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.size()

        # (batch, seq_len, d_model) → (batch, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)

        # (batch, seq_len, num_heads, d_k) → (batch, num_heads, seq_len, d_k)
        # transpose로 num_heads를 앞으로 이동
        # 이렇게 하면 각 헤드에서 독립적으로 Attention 계산 가능
        return x.transpose(1, 2)

    def merge_heads(self, x):
        """
        여러 헤드를 다시 합침

        Args:
            x: (batch, num_heads, seq_len, d_k)

        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.size()

        # (batch, num_heads, seq_len, d_k) → (batch, seq_len, num_heads, d_k)
        x = x.transpose(1, 2)

        # (batch, seq_len, num_heads, d_k) → (batch, seq_len, d_model)
        # contiguous(): 메모리를 연속적으로 재배열 (view 사용 전 필요)
        return x.contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, query, key, value, mask=None):
        """
        Multi-Head Attention 순전파

        Args:
            query: (batch, seq_len, d_model)
            key: (batch, seq_len, d_model)
            value: (batch, seq_len, d_model)
            mask: 마스크 텐서 (선택사항)

        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        # 1단계: Linear 변환으로 Q, K, V 생성
        Q = self.W_Q(query)  # (batch, seq_len, d_model)
        K = self.W_K(key)
        V = self.W_V(value)

        # 2단계: 헤드로 분할
        Q = self.split_heads(Q)  # (batch, num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 3단계: Scaled Dot-Product Attention
        # 각 헤드에서 독립적으로 Attention 수행
        attn_output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask
        )
        # attn_output: (batch, num_heads, seq_len, d_k)

        # 4단계: 헤드 합치기 (Concat)
        output = self.merge_heads(attn_output)  # (batch, seq_len, d_model)

        # 5단계: 최종 Linear 변환
        output = self.W_O(output)  # (batch, seq_len, d_model)

        return output, attention_weights


# 테스트
print("\nMulti-Head Attention 테스트:")
print("-" * 40)

# 하이퍼파라미터
d_model = 512
num_heads = 8
batch_size = 2
seq_len = 10

print(f"설정:")
print(f"  d_model (전체 차원): {d_model}")
print(f"  num_heads (헤드 수): {num_heads}")
print(f"  d_k (헤드당 차원): {d_model // num_heads}")
print(f"  batch_size: {batch_size}")
print(f"  seq_len: {seq_len}")

# 모델 생성
mha = MultiHeadAttention(d_model, num_heads)

# 랜덤 입력 생성
x = torch.randn(batch_size, seq_len, d_model)

# Self-Attention: Q, K, V 모두 같은 입력
output, attention_weights = mha(x, x, x)

print(f"\n입력 형태: {x.shape}")
print(f"출력 형태: {output.shape}")
print(f"Attention 가중치 형태: {attention_weights.shape}")
print(f"  → (batch, num_heads, seq_len, seq_len)")
print(f"  → 각 헤드마다 (seq_len × seq_len) 가중치 행렬")

print("\n" + "=" * 60)
print("6. 단일 헤드 vs Multi-Head 비교")
print("=" * 60)


class SingleHeadAttention(nn.Module):
    """비교를 위한 단일 헤드 Attention"""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        Q = self.W_Q(query)
        K = self.W_K(key)
        V = self.W_V(value)

        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(d_k, dtype=torch.float32)
        )

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        return output, attention_weights


# 파라미터 수 비교
single_head = SingleHeadAttention(d_model)
multi_head = MultiHeadAttention(d_model, num_heads)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


print(f"\n파라미터 수 비교:")
print(f"  단일 헤드: {count_parameters(single_head):,}")
print(f"  Multi-Head ({num_heads}개): {count_parameters(multi_head):,}")
print(f"\n→ Multi-Head는 W_O가 추가되어 약간 더 많지만,")
print(f"  Attention 자체의 파라미터 수는 동일합니다!")

# Attention 가중치 비교
print(f"\nAttention 가중치 비교:")
_, single_weights = single_head(x, x, x)
_, multi_weights = multi_head(x, x, x)

print(f"  단일 헤드: {single_weights.shape}")
print(f"    → 하나의 (seq_len × seq_len) 가중치 행렬")
print(f"  Multi-Head: {multi_weights.shape}")
print(f"    → {num_heads}개의 (seq_len × seq_len) 가중치 행렬")
print(f"    → 각 헤드가 서로 다른 관계 패턴을 학습!")

print("\n" + "=" * 60)
print("7. 각 헤드는 무엇을 학습하는가?")
print("=" * 60)

# ============================================================
# 실제 학습된 Transformer를 분석해보면:
# - Head 1: 바로 다음 단어에 집중 (local attention)
# - Head 2: 문장의 첫 단어에 집중 (global attention)
# - Head 3: 같은 품사의 단어에 집중 (syntactic attention)
# - Head 4: 의미적으로 관련된 단어에 집중 (semantic attention)
# - ...
#
# 각 헤드가 자동으로 서로 다른 역할을 학습합니다!
# 이것이 Multi-Head의 힘입니다.
# ============================================================

print("""
실제 학습된 Transformer의 헤드들이 학습하는 것:

Head 1: "바로 옆 단어" 패턴
        ┌───┬───┬───┬───┬───┐
        │ ■ │   │   │   │   │  The
        │   │ ■ │   │   │   │  cat
        │   │   │ ■ │   │   │  sat
        │   │   │   │ ■ │   │  on
        │   │   │   │   │ ■ │  mat
        └───┴───┴───┴───┴───┘

Head 2: "문장 시작" 패턴
        ┌───┬───┬───┬───┬───┐
        │ ■ │   │   │   │   │  The
        │ ■ │   │   │   │   │  cat
        │ ■ │   │   │   │   │  sat
        │ ■ │   │   │   │   │  on
        │ ■ │   │   │   │   │  mat
        └───┴───┴───┴───┴───┘

Head 3: "명사끼리" 패턴
        ┌───┬───┬───┬───┬───┐
        │   │   │   │   │   │  The
        │   │ ■ │   │   │ ■ │  cat
        │   │   │   │   │   │  sat
        │   │   │   │   │   │  on
        │   │ ■ │   │   │ ■ │  mat
        └───┴───┴───┴───┴───┘

각 헤드가 자동으로 서로 다른 관계를 전문화합니다!
(이것은 학습을 통해 자연스럽게 발생)
""")

print("\n" + "=" * 60)
print("8. PyTorch nn.MultiheadAttention과 비교")
print("=" * 60)

# PyTorch 내장 Multi-Head Attention
pytorch_mha = nn.MultiheadAttention(
    embed_dim=d_model,
    num_heads=num_heads,
    batch_first=True  # 입력 형태를 (batch, seq, embed)로 설정
)

# 동일한 입력으로 테스트
pytorch_output, pytorch_weights = pytorch_mha(x, x, x)

print(f"PyTorch 내장 nn.MultiheadAttention:")
print(f"  입력: {x.shape}")
print(f"  출력: {pytorch_output.shape}")
print(f"  가중치: {pytorch_weights.shape}")
print(f"\n우리가 구현한 것과 동일한 구조입니다!")
print(f"(내부 구현 최적화로 속도는 더 빠름)")

print("\n" + "=" * 60)
print("핵심 요약")
print("=" * 60)
print("""
1. 왜 Multi-Head인가?
   - 단일 헤드는 하나의 관계 패턴만 잘 포착
   - 여러 헤드로 다양한 관계를 동시에 학습

2. 구조:
   MultiHead(Q,K,V) = Concat(head_1, ..., head_h) × W_O
   - 각 헤드: 독립적인 Attention 수행
   - Concat 후 W_O로 합침

3. 차원:
   - d_model = num_heads × d_k
   - 예: 512 = 8 × 64
   - 총 파라미터 수는 단일 헤드와 비슷!

4. 핵심 메서드:
   - split_heads: (batch, seq, d_model) → (batch, heads, seq, d_k)
   - merge_heads: (batch, heads, seq, d_k) → (batch, seq, d_model)

5. 각 헤드의 역할:
   - 학습을 통해 자동으로 전문화
   - 문법적, 의미적, 위치적 관계 등 분담

다음 단계에서는 Positional Encoding을 배웁니다!
(Attention은 순서를 모르기 때문에, 위치 정보 추가 필요)
""")
