# Causal Effect

- 진행일: 250829
- 발표자: 한지수

## 사전 학습

### Causal Inference

- 정의: 단순한 상관관계인 correlation 이 아니라, 변수 간의 원인과 결과 cause-effect 구조를 추론하는 방법론
    - correlation: 두 변수가 같이 변함
    - cause-effect: 한 변수가 바뀌면, 다른 변수가 이에 따라 변함

- 예시:
    - 아이스크림 판매량과 익사 사고 건수: 두 값이 여름에 같이 올라감. 날씨라는 교란변수 (Confounder)에 의한 것이기 원인과 결과의 관계가 아님.
    - 약 복용과 완치: 환자가 약을 먹어서 나았는지 시간이 지나서 나았는지 인과효과가 명확하지 않을 수 있음

- 수학적 표현:
    - $do(\cdot)$: 단순한 관찰이 아니라 실제로 변수를 조작했을 때의 효과
    - $\mathbb{E}[Y|do(X=1)]$: 모두 약을 먹었을 때의 평균 효과
    - $\mathbb{E}[Y|do(X=0)]$: 아무도 약을 안 먹었을 때의 평균 효과
$$
\text{Causal Effect} = \mathbb{E}[Y|do(X=1)]-\mathbb{E}[Y|do(X=0)]
$$

- 핵심: $\text{CE}$는 해당 변수의 인과 효과만을 드러내는 것으로 상관관계가 아니라 인과적 추론관계를 수치화하는 것이다.

### 딥러닝에서의 필요성

- 딥러닝 모델은 데이터를 그대로 학습하기에 데이터, 모델의 bias에 취약하다.
    - VQA (Visual Question Answering)에서 Vision이 약화되고, Question만 보고 답을 맞추는 bias가 발생한다.
    - 불균형 데이터셋에서 다수 클래스에 bias 되어 학습되는 경우가 발생한다.

- 이에 따라 Causal Inference의 개념을 도입하여 진짜 원인적 요인만 반영하여 모델을 학습하도록 하는 기술이 필요하다.


### 용어 정리

- 독립변인 (Independent Variable):
    - 정의: 결과에 영항을 미치는 원인의 역할을 하는 변수
    - 예시: 모델의 입력 $X$

- 종속변인 (Dependent Variable):
    - 정의: 원인에 의해 변화하는 결과의 역할을 하는 변수
    - 예시: 모델의 출력과 최종 레이블 $Y$

- 교란변인 (Confounder):
    - 정의: $X$와 $Y$에 모두 영향을 미쳐, 인과관계를 왜곡하는 변수
    - 예시: VQA에서 질문만 보고 답을 맞추는 언어적 편향이 Confounder 이다

- 매개변인 (Mediator):
    - 정의: $X$가 $Y$의 원인일 때, 중간 경로에서 작동하는 변수
    - 예시: 중간 feature (latent representation) 등

- 조작변인 (Manipulation Variable):
    - 정의: 실험자가 직접 조작하여 변화시키는 변수 (실제 인과효과를 확인하는 핵심)
    - 예시: 데이터 증강 (모델이 편향되지 않고 학습하는지 확인 가능), Counterfactual learning (입력이 특정 요소를 의도적으로 바꿔서 모델의 반응 측정)

- 통제변인 (Control Variable):
    - 정의: 분석 시 결과에 영향을 줄 수 있는 외부 요인을 고정하거나 통제해서 순수 인과효과를 추정
    - 예시: Domain Adaptation에서 Domain을 통제하여 편향을 제거하고 순수한 feature-label의 관계를 추정

- 예시: “운동이 체중 감량에 영향을 주는가”
    - 독립변인: 운동 여부
    - 종속변인: 체중 감량 정도
    - 교란변인: 식습관, 유전적 체질
    - 매개변인: 체지방 감소
    - 조작변인: 실험 설계에서 운동량을 강제로 조정
    - 통제변인: 실험 시 식습관을 동일하게 맞춤

### Counterfactual VQA: A Cause-Effect Look at Language Bias (CVPR 2021)

- 문제: VQA (Visual Question Answering) 모델이 언어적 편향 문제를 가짐
- 예시: V가 초록색 바나나이고, Q가 "mostly yellow, seldom green" 일 때, language에 편향된 답을 생성하여 yellow를 정답으로 생성하는 문제 발생
- 해결 방법: 기존 출력 분포와 이미지를 block하고 Q만 입력할 때의 출력 분포의 차를 구하여 language bias가 제거되고, 이미지가 실제로 답변에 기여한 출력 분포를 파악한다.
- 요약: 이미지가 실제로 기여한 Conterfactual 효과를 계산한다. 즉, causal effect를 정의하여 해당 값을 기반으로 최종 출력 분포를 조정한다. 이를 통해 language bias 문제를 해결한다.

### Debiased Learning from Naturally Imbalanced Pseudo-Labels (CVPR 2022)

- 문제: 반지도 학습에서 모델 출력이 bias 되어 Pseudo-Label은 자연적으로 클래스 불균형을 가진다.
- 해결방법: 다음의 수식과 같이 많이 예측되는 class의 logit을 줄여서 bias를 완화한다.
    - $f(\alpha(x_i))$: weak aumentation 된 데이터에 대한 logit
    - $\hat{p}$: 전체 unlabeled 데이터에 대한 moving average 분포
    - $\lambda$: debias 정도

$$
\tilde{f}_i=f(\alpha(x_i))-\lambda\log\hat{p}\\
\hat{p}\leftarrow m\hat{p}+(1-m)\frac{1}{\mu B}\sum_{k=1}^{\mu B}p_k
$$



### CDMAD: Class-Distribution-Mismatch-Aware Debiasing for Class-Imbalanced Semi-Supervised Learning (CVPR 2024)

- 문제: 반지도 학습에서 데이터셋의 클래스 불균형 문제 발생
    - 클래스 불균형에 의해 분류기 자체가 다수 클래스에 편향됨
    - 편향된 Pseudo-Label이 학습에 이용됨
- 해결 방법: 
    - 다음의 수식과 같이 분류기의 클래스 편향을 반영하여 모델의 예측을 수정함
    - $g_{\theta}(\mathcal{I})$: 패턴이 없는 단색 이미지에서 측정한 로짓으로 분류기의 클래스 편향성을 나타낸다

$$
g^{*}_{\theta}(x^{test}) = g_{\theta}(x^{test})-g_{\theta}(\mathcal{I})
$$

