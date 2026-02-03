# Transformer 학습 프로젝트

PyTorch로 Transformer 아키텍처를 처음부터 구현하며 원리를 이해하는 학습 프로젝트입니다.

## 학습 목표

- Transformer의 핵심 개념을 **직접 구현**하며 이해
- 단계별로 복잡도를 높여가며 학습
- 최종적으로 간단한 태스크에 적용

## 진행 상황

- [x] 1단계: PyTorch 기초
- [x] 2단계: Attention 메커니즘
- [ ] 3단계: Multi-Head Attention
- [ ] 4단계: Positional Encoding
- [ ] 5단계: Transformer 블록
- [ ] 6단계: 전체 Transformer
- [ ] 7단계: 문자열 복사 태스크
- [ ] 8단계: 숫자→영어 변환

## 학습 단계

### 1단계: PyTorch 기초 (`01_pytorch_basics.py`)
**목표**: Transformer 구현에 필요한 PyTorch 기본기 습득
- 텐서 생성 및 연산
- 자동 미분 (autograd)
- nn.Module로 신경망 만들기
- 손실함수와 옵티마이저
- 학습 루프: `zero_grad() → forward → loss → backward() → step()`

### 2단계: Attention 메커니즘 (`02_attention_basics.py`)
**목표**: Transformer의 핵심인 Attention 개념 이해 및 구현
- Query, Key, Value 개념 (도서관 검색 비유)
- Scaled Dot-Product Attention: `softmax(QK^T / √d_k) × V`
- Self-Attention 구현
- Attention 가중치 시각화

### 3단계: Multi-Head Attention (`03_multi_head_attention.py`)
**목표**: 여러 관점에서 Attention을 수행하는 방법 이해
- 단일 헤드 vs 멀티 헤드 비교
- Multi-Head Attention 직접 구현
- 파라미터 차원 이해 (d_model, num_heads, d_k)

### 4단계: Positional Encoding (`04_positional_encoding.py`)
**목표**: 순서 정보를 인코딩하는 방법 이해
- 왜 위치 정보가 필요한가? (Attention은 순서를 모름)
- 사인/코사인 기반 인코딩 구현
- 시각화로 패턴 이해

### 5단계: Transformer 블록 (`05_transformer_block.py`)
**목표**: 개별 컴포넌트를 조합하여 Transformer 블록 완성
- Layer Normalization
- Feed Forward Network
- Residual Connection
- Encoder 블록 조립
- Decoder 블록 조립 (Masked Attention 포함)

### 6단계: 전체 Transformer (`06_full_transformer.py`)
**목표**: 완전한 Transformer 모델 구현
- Encoder 스택
- Decoder 스택
- 최종 출력 레이어

### 7단계: 문자열 복사 태스크 (`07_copy_task.py`)
**목표**: 구현한 Transformer로 가장 단순한 문제 해결
- 랜덤 문자열 데이터 생성
- 입력을 그대로 출력하는 복사 태스크
- Attention 가중치 시각화로 동작 확인

### 8단계: 숫자→영어 변환 (`08_number_to_english.py`)
**목표**: 조금 더 복잡한 변환 태스크 적용
- 숫자를 영어로 변환 (1 → one, 23 → twenty three)
- 의미 있는 변환 학습
- 모델 성능 평가

## 실행 방법

```bash
# 의존성 설치
pip3 install -r requirements.txt

# 각 단계 실행
python3 01_pytorch_basics.py
python3 02_attention_basics.py
# ...
```

## 참고 자료

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 원본 논문
- [PyTorch 공식 문서](https://pytorch.org/docs/)
