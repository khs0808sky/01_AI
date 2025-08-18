# 01_AI

## 📅 목차

- [2025-08-11](#2025-08-11)
- [2025-08-12](#2025-08-12)
- [2025-08-13](#2025-08-13)
- [2025-08-14](#2025-08-14)
- [2025-08-18](#2025-08-18)

<br><br><br>

---

## **2025-08-11**

---

### 평균제곱 오차(MSE, Mean Squared Error)

* **정의**: 예측값과 실제값의 차이를 제곱한 뒤, 그 평균을 계산한 값.
* **공식**:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

* $y_i$: 실제값
* $\hat{y}_i$: 예측값
* $n$: 데이터 개수
* **특징**

  * 오차를 제곱하므로 양수화되고, 큰 오차에 더 큰 패널티를 부여.
  * 회귀 모델의 성능 지표로 자주 사용됨.
* **단점**

  * 이상치(outlier)에 민감함.

---

### 경사하강법(Gradient Descent)

* **정의**: 비용 함수(cost function)를 최소화하기 위해 매개변수를 반복적으로 조정하는 최적화 알고리즘.
* **아이디어**:

  * 비용 함수의 기울기(gradient)를 계산해, 가장 가파르게 내려가는 방향으로 매개변수를 업데이트.
* **공식**:

$$
\theta = \theta - \alpha \cdot \frac{\partial J(\theta)}{\partial \theta}
$$

* $\theta$: 매개변수(가중치 등)
* $\alpha$: 학습률(learning rate)
* $J(\theta)$: 비용 함수
* **종류**

  1. **배치 경사하강법(Batch GD)**: 모든 데이터 사용 → 안정적, 하지만 느림.
  2. **확률적 경사하강법(SGD)**: 데이터 1개씩 사용 → 빠르지만 변동이 큼.
  3. **미니배치 경사하강법(Mini-batch GD)**: 일정 크기 데이터 묶음 사용 → 속도와 안정성 균형.
* **학습률 주의사항**

  * 너무 크면 발산(oscillation)할 수 있음.
  * 너무 작으면 학습이 느려짐.

---

📅[목차로 돌아가기](#-목차)

---

## **2025-08-12**

---

**퍼셉트론(Perceptron)**

* 인공신경망의 가장 기본적인 형태로, 단일 층에서 입력값에 가중치를 곱하고 합산 후 활성화 함수를 통해 출력을 결정.
* 동작 원리:

  1. 입력 $x_1, x_2, \dots, x_n$과 가중치 $w_1, w_2, \dots, w_n$을 곱해 합산.
  2. 바이어스 $b$를 더함.
  3. 활성화 함수(예: 계단 함수)로 결과를 변환해 출력.
* 단점: XOR 문제처럼 선형 분리가 불가능한 문제는 해결 불가 → 다층 퍼셉트론(MLP) 필요.

---

**오차 역전파(Backpropagation)**

* 다층 신경망에서 학습 시, 출력층에서 계산한 오차를 은닉층으로 거꾸로 전파해 각 가중치를 업데이트하는 알고리즘.
* 과정:

  1. **순전파**: 입력 → 은닉층 → 출력층 계산.
  2. **오차 계산**: 예측값과 실제값 비교 후 손실 함수로 오차 구함.
  3. **역전파**: 오차를 각 층으로 전달하며 기울기 계산(연쇄법칙 이용).
  4. **가중치 업데이트**: 경사하강법 기반으로 조정.
* 장점: 효율적으로 다층 네트워크 학습 가능.

---

**고급 경사하강법(Advanced Gradient Descent)**

* 기본 경사하강법의 한계를 개선한 최적화 기법.
* 주요 기법:

  * **모멘텀(Momentum)**: 이전 기울기를 누적해 관성처럼 적용, 진동 완화 및 빠른 수렴.
  * **AdaGrad**: 각 파라미터별로 학습률 조정, 드문 업데이트가 필요한 파라미터에 큰 학습률 부여.
  * **RMSProp**: 기울기의 제곱 평균을 사용해 학습률 조절, AdaGrad의 학습률 급감 문제 개선.
  * **Adam**: 모멘텀 + RMSProp 결합, 현재 가장 널리 쓰이는 최적화 알고리즘.

---

**원-핫 인코딩(One-Hot Encoding)**

* 범주형 데이터를 컴퓨터가 이해할 수 있는 벡터 형태로 변환하는 방법.
* 원리: N개의 범주 중 해당되는 위치만 1, 나머지는 0으로 표시.
* 예시: {고양이, 강아지, 토끼}에서 "강아지" → \[0, 1, 0].
* 장점: 범주형 변수를 수치화하면서 순서 정보가 포함되지 않음.
* 단점: 범주 수가 많으면 벡터가 매우 커짐(희소 행렬 문제).

---

**학습셋과 테스트셋 구분**

* **학습셋(Training Set)**: 모델을 학습시키는 데 사용되는 데이터.
* **검증셋(Validation Set)**: 하이퍼파라미터 조정 및 과적합 방지에 사용.
* **테스트셋(Test Set)**: 최종적으로 모델의 일반화 성능을 평가하는 데이터.
* 데이터 분리 예시: 일반적으로 6:2:2 또는 8:1:1 비율로 나눔.
* 주의: 테스트셋은 학습 과정에 절대 사용하지 않아야 함(데이터 누수 방지).

---

📅[목차로 돌아가기](#-목차)

---
---

## **2025-08-13**

---

**데이터의 확인과 검증셋**

* **데이터 확인(Data Inspection)**: 학습 전에 데이터의 분포, 결측치, 이상치 등을 점검.
* **검증셋(Validation Set)**: 학습 과정에서 모델 성능을 평가하고 하이퍼파라미터를 조정하기 위해 사용.

  * 학습셋과 별도로 분리하여 모델이 새로운 데이터에 얼마나 잘 일반화되는지 확인.

---

**과적합(Overfitting)과 자동중단(Early Stopping)**

* **과적합**: 모델이 학습 데이터에 지나치게 맞춰져서 새로운 데이터에 대한 성능이 떨어지는 현상.
* **자동중단(Early Stopping)**: 검증셋 성능이 개선되지 않으면 학습을 조기 종료하여 과적합 방지.

  * 일반적으로 검증 손실이 연속적으로 증가하면 학습 중단.

---

**컨볼루션 신경망(CNN, Convolutional Neural Network)**

* 이미지, 영상 등의 공간적 데이터를 처리하기 위해 설계된 신경망.
* 핵심 요소:

  1. **컨볼루션 레이어**: 필터(커널)를 통해 특징 맵(feature map) 생성.
  2. **맥스 풀링(Max Pooling)**: 공간 크기를 줄이고, 주요 특징만 추출.
  3. **드롭아웃(Dropout)**: 일부 뉴런을 무작위로 비활성화하여 과적합 방지.
  4. **플래튼(Flatten)**: 다차원 데이터를 1차원 벡터로 변환해 완전 연결층(FC)에 입력.

---

**NLP(자연어 처리, Natural Language Processing)**

* 컴퓨터가 인간의 언어를 이해하고 처리할 수 있도록 하는 기술.
* 주요 작업: 텍스트 분류, 감성 분석, 기계 번역, 챗봇 등.

---

**임베딩(Embedding)**

* 단어, 문장, 아이템 등 범주형 데이터를 실수 벡터로 변환하는 방법.
* 특징:

  * **밀집 표현(Dense Representation)**: 희소 표현(one-hot)보다 효율적.
  * 의미적 유사성을 벡터 거리로 측정 가능.
* 예시:

  * Word2Vec, GloVe: 단어 수준 임베딩.
  * BERT, GPT 계열: 문맥을 반영한 문장 임베딩.

---

📅[목차로 돌아가기](#-목차)

---

## **2025-08-14**

---

### Transformers 한눈에 보기

* **핵심 아이디어**: 순서를 따라가며 정보를 전달하던 RNN 대신, **Self-Attention**으로 입력 전체를 한 번에 바라보며 토큰 간 의존성을 학습. 병렬처리로 학습/추론이 빠르고, 장거리 의존성에 강함.
* **대표 구조**: (1) **Encoder–Decoder**(번역 등, 예: T5), (2) **Encoder-only**(이해/분류, 예: BERT), (3) **Decoder-only**(생성, 예: GPT).

---

### 블록 구성 요소(한 층)

1. **임베딩 + 위치정보**

   * 토큰을 실수 벡터로 변환(**Token Embedding**) 후, 순서 정보를 주입(**Positional Encoding**: Sinusoidal/학습형/**RoPE**/ALiBi 등).

2. **멀티헤드 Self-Attention (MHSA)**

   * 각 토큰에서 **Query(Q)**, **Key(K)**, \*\*Value(V)\*\*를 만들고 유사도를 가중합:
     
   ![attention](https://latex.codecogs.com/svg.image?\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+\text{mask}\right)V)

   * 여러 **head**(보통 8, 12, 16…)로 서로 다른 서브공간의 패턴을 병렬 학습한 뒤 concat → 선형변환.

3. **포지션별 피드포워드(FFN)**

   * 각 토큰 벡터에 독립 적용되는 2층 MLP (차원 일반적으로 $4\times d_{model}$), 활성함수 **GELU** 등.

4. **잔차 연결(Residual) + LayerNorm**

   * 안정적 학습을 위해 `x + Sublayer(x)` 후 **LayerNorm**. (현대 모델은 **Pre-LN**가 일반적)

---

### 마스크(Mask)

* **Padding Mask**: 패딩 토큰 점수 차단.
* **Causal(look-ahead) Mask**: 디코더에서 미래 토큰을 보지 못하게 차단(생성 모델 필수).
* **Cross-Attention**: 디코더가 인코더 출력을 참조할 때 사용(번역 등).

---

### 세 가지 아키텍처와 활용

* **Encoder–Decoder**: 입력 인코딩 → 디코더가 **Cross-Attention**으로 참조하며 출력 생성(기계번역, 요약).
* **Encoder-only (BERT류)**: **Masked Language Modeling**으로 문맥 이해에 특화(분류/추출/문서 이해).
* **Decoder-only (GPT류)**: **Autoregressive** 예측으로 자연스러운 텍스트/코드/대화 생성.

---

### 학습 목표(Objectives)

* **Autoregressive**: 다음 토큰 예측(언어 생성).
* **Masked LM**: 일부 토큰 가리기 → 맞히기(언어 이해).
* **Sequence-to-Sequence**: 입력 시퀀스 → 출력 시퀀스 매핑(번역/요약).

---

### 토크나이징

* **BPE / WordPiece / SentencePiece**: 서브워드 단위로 분해(희귀어/신조어에 강함).
* **스페셜 토큰**: BOS/EOS/PAD/CLS/SEP 등.

---

### 디코딩 전략(생성 시)

* **Greedy**, **Beam Search**(정밀), **Top-k**, **Top-p(누클리어스)**, **Temperature**(창의성 제어).

  * 일반적 조합: Top-p + Temperature, 또는 Beam(요약/번역).

---

### 계산 복잡도와 효율화

* **표준 Self-Attention**: $O(n^2)$ 메모리/시간(문맥 길이 $n$).
* 효율 기법: **FlashAttention**(메모리 효율), **Sparse/Sliding-window**(Longformer), **Linear Attention**(Performer/Kernel), **압축/리샘플링**, **KV 캐시**(디코딩 가속).

---

### 위치 인코딩/바이어스

* **Sinusoidal**(고정), **Learned**(학습형), **RoPE**(회전형, 롱컨텍스트/상대 위치에 유리), **ALiBi**(길이 일반화).
* 긴 문맥 모델은 RoPE 스케일링/ALiBi/윈도우화로 안정성 확보.

---

### 최적화 & 학습 팁

* **AdamW**(가중치 감쇠), **Warmup + Cosine/Linear decay** 스케줄.
* **Mixed Precision(FP16/BF16)**, **Gradient Accumulation/Checkpointing**.
* **정규화/안정화**: Dropout, Weight Tying, Norm 튜닝.
* **파인튜닝**: 전체 미세조정, **LoRA/Adapter**(경량), Prompt/Prefix Tuning(매개변수 적음).
* **정렬(Alignment)**: RLHF/DPO 등(대화 품질/안전성 향상, 필요 시).

---

### 비전/오디오/멀티모달

* **ViT**: 이미지를 패치로 쪼개 토큰처럼 처리(CLS 토큰을 통해 분류).
* **멀티모달**: 텍스트+이미지/음성 등을 하나의 토큰 시퀀스로 결합하여 학습.

---

### 한계와 주의점

* **메모리/시간 비용**: $O(n^2)$로 길이가 길어질수록 급증.
* **환각(Hallucination)**, **길이 편향**, **노이즈 민감성**.
* 데이터 누수/편향 관리, 평가셋 엄격 분리 필수.

---

### 핵심 하이퍼파라미터 빠르게 정리

* $d_{model}$ (임베딩 차원), **heads**(멀티헤드 수), **layers**(층 수), **FFN 차원**($\approx 4\times d_{model}$), **context length**, **dropout**, **lr/warmup**.

---

📅[목차로 돌아가기](#-목차)

---

---

## **2025-08-18**

---

### 1) 검증셋(Validation Set)

* **목적**: 모델이 학습 데이터에만 과적합(overfitting)되지 않았는지 확인하고, **하이퍼파라미터**(학습률, 레이어 수, 드롭아웃 비율 등)를 조정하는 데 사용.
* **구성**:

  * 데이터셋 → **Train / Validation / Test**로 분리.
  * Train: 가중치 학습.
  * Validation: 중간 점검(성능 측정, Early Stopping).
  * Test: 최종 성능 평가(실전 배치 전).
* **사용 예시**:

  * **Early Stopping**: Validation loss가 일정 횟수 이상 개선되지 않으면 학습 종료.
  * 모델 선택(Model Selection): Validation 성능이 가장 좋은 checkpoint를 선택.

---

### 2) Word Embedding

* **정의**: 단어를 고차원 희소 벡터(One-hot)가 아닌, **의미적 관계를 반영하는 저차원 연속 벡터 공간**에 매핑하는 방법.
* **대표 방법**:

  1. **Word2Vec (CBOW / Skip-gram)**

     * CBOW: 주변 단어 → 중앙 단어 예측.
     * Skip-gram: 중앙 단어 → 주변 단어 예측.
     * 단어 간 **코사인 유사도**로 의미 관계 포착.
  2. **GloVe**: 말뭉치 전체의 **공동 등장 확률 통계**를 활용해 학습.
  3. **FastText**: 단어 내부 **n-gram** 단위까지 고려 → 신조어, 희귀어 대응 가능.
  4. **Contextual Embedding (ELMo, BERT, GPT 등)**

     * 동일한 단어라도 문맥에 따라 다른 벡터 제공.
     * 현대 NLP에서는 주로 이 방식 사용.
* **특징**:

  * 유사 단어는 벡터 공간에서 가깝게 위치.
  * “king – man + woman ≈ queen” 같은 벡터 연산 가능.

---

### 3) Attention Value

* **Self-Attention 메커니즘**에서 Q, K, V 벡터를 통해 계산.

  * $Q = XW^Q,\ K = XW^K,\ V = XW^V$
  * 유사도: $\text{score}(Q,K) = \frac{QK^\top}{\sqrt{d_k}}$
  * 가중치: $\alpha = \text{softmax}(\text{score})$
  * 출력: $\text{Attention}(Q,K,V) = \alpha V$
* **Value(V)의 역할**:

  * K와 Q로 "어디를 집중할지" 결정한 후,
  * V는 그 위치의 \*\*실제 정보(콘텐츠)\*\*를 담고 있음.
* **직관적 비유**:

  * Q = 내가 지금 궁금한 질문
  * K = 모든 정보들의 “색인/주소”
  * V = 색인이 가리키는 “실제 내용”
    → 결국 Attention Value는 최종적으로 모델이 가져오는 **콘텐츠 정보**.

---

### 4) 파인튜닝(Fine-Tuning)

* **정의**: 사전 학습(Pre-training)된 모델을 특정 태스크에 맞게 추가 학습하는 과정.
* **종류**:

  1. **Full Fine-Tuning**

     * 모델 전체 파라미터를 학습 데이터에 맞게 다시 학습.
     * 성능 좋지만 비용 크고, Catastrophic Forgetting 위험.
  2. **Feature Extraction**

     * 사전학습 모델은 고정, 상단에 작은 분류기(MLP, Linear)만 학습.
  3. **Parameter-Efficient Fine-Tuning (PEFT)**

     * 일부 파라미터만 조정 → 효율성↑
     * 예: **LoRA**, **Adapter**, **Prefix Tuning**.
* **활용 예시**:

  * BERT → 감정 분류, 문장 유사도, NER 등.
  * GPT → 대화, 코드 생성, 특정 도메인 언어.
* **실무 팁**:

  * 데이터 적을 땐 PEFT 기법 활용 추천.
  * 과적합 방지를 위해 Early Stopping/Dropout 필요.
  * 학습률은 보통 Pre-training보다 훨씬 작게 설정(예: 1e-5\~5e-5).

---

📅[목차로 돌아가기](#-목차)

---

