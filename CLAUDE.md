# Claude 컨텍스트

이 파일은 Claude가 세션 시작 시 프로젝트 컨텍스트를 이해하는 데 사용됩니다.

## 사용자 정보

- **경험 수준**: Python은 알지만 딥러닝은 입문자
- **학습 스타일**: 원리를 이해하며 직접 구현
- **선호**: 한글 주석 풍부하게, 단계별 설명

## 환경 설정

- **Python 실행**: `python3` 사용 (pip도 `pip3`)
- **프레임워크**: PyTorch

## 현재 진행 상황

### 완료된 단계
- **1단계: PyTorch 기초** - 완료, 커밋됨

### 현재 진행 중
- **2단계: Attention 메커니즘** - 코드 작성 완료, 검토 중

### 다음 단계
- 3단계: Multi-Head Attention

## 학습 중 이해한 핵심 개념

### 1단계에서 배운 것
- **텐서**: 다차원 배열, `requires_grad=True`로 기울기 추적
- **자동 미분**: `backward()`는 스칼라에서 호출, 기울기는 `requires_grad=True`인 텐서에만 계산
- **학습 루프**: `zero_grad() → forward → loss → backward() → step()`
- **PyTorch 설계 철학**: 상태 변경 기반, 함수형이 아님
- **optimizer**: 파라미터 참조를 보유하고 `step()`에서 업데이트 수행

### 2단계에서 배울 것
- Query, Key, Value 개념
- Scaled Dot-Product Attention
- Self-Attention

## 주의사항

- 손실함수에서 `backward()` 호출 → 모델 파라미터의 `.grad`에 기울기 저장
- 입력 데이터(X)는 `requires_grad=False`, 가중치(W)만 `True`
- `zero_grad()` 필수 (기울기 누적 방지)

## 학습 진행 방식

1. 코드 작성 및 실행
2. 사용자 질문 → 개념 설명
3. 이해 확인 후 커밋
4. 다음 단계 진행

## 전체 학습 계획 (8단계)

### 1단계: PyTorch 기초 (`01_pytorch_basics.py`) ✅ 완료
**목표**: Transformer 구현에 필요한 PyTorch 기본기 습득
- 텐서 생성 및 연산
- 자동 미분 (autograd)
- nn.Module로 신경망 만들기
- 손실함수와 옵티마이저
- 학습 루프: `zero_grad() → forward → loss → backward() → step()`

### 2단계: Attention 메커니즘 (`02_attention_basics.py`) 🔄 진행중
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
