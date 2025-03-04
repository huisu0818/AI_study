# CS229: Chapter2

- Classification and Logistic Regression
- Lecture 3

## Study Members
- Huisu Lee (Lecture Note Summary)
    - Ajou University, Software and Computer Engineering
    - Computer Vision, Deep Learning

- Dogyu Lee (Code Implementation)
    - Seoul National University, Electrical and Computer Engineering
    - Deep Learning Hardware Acceleration, Optimization

## Contents
- Logistic Regression
- Learning Algorithm
- Newton's Method

## Logistic Regression

### Motivation
분류 문제에서는 출력값 $y$ 가 몇 개의 특정한 값만 가질 수 있다. 가장 단순한 Binary Classification 를 다뤄보자.

이진 분류 문제에서 $y\in\{0, 1\}$두 개의 값 중 하나를 가진다. 
- $y=1$: Positive Class (ex: malignant tumor)
- $y=0$: Negative Class (ex: benign tumor)

이를 Linear Regression으로 아래의 그림과 같이 접근해보자.

<div style="text-align: center;">
    <img src="./Fig1.png" alt="nn" width="500">
</div>

이럴 경우 다음의 두가지 문제가 발생한다.

- $h_\theta(x)>1$ or $h_\theta(x)<0$인 경우가 발생
- $h_\theta(x)$를 "해당 샘플이 클래스 $y=1$일 확률"로 해석할 수 없음

### Hypothesis

이러한 문제를 해결하기 위해 다음의 Sigmoid Function $g(z)$를 도입하자.

$$
g(z) = \frac{1}{1+e^{-z}}
$$

$g(z)$는 다음의 그래프로 표현되며 아래의 특징을 갖는다.

<div style="text-align: center;">
    <img src="./Fig2.png" alt="nn" width="500">
</div><br>

- $z\rightarrow\infty: g(z)\rightarrow1$
- $z\rightarrow-\infty: g(z)\rightarrow0$
- $0 < g(z) <1$

이제 Sigmoid Function $g(z)$로 다음과 같이 hypothesis $h_\theta(x)$를 정의한다.

$$
h_\theta(x) = g(\theta^Tx)=\frac{1}{1+e^{-\theta^Tx}}
$$

이렇게 정의함에 따라 아래의 사진과 같이 $x$에 대한 예측을 클래스 $y=1$ 혹은 $y=0$에 대한 확률로 이해할 수 있다.

<div style="text-align: center;">
    <img src="./Fig3.png" alt="nn" width="500">
</div><br>

이렇게 정의된 $h_\theta(x)$를 해석하여 Dicision Boundary에 대해 이해해보자.

Logistic Regression에서 예측 class는 다음과 같이 정의된다.

$$
y=
\begin{cases}
1 & \text{if}~h_\theta(x) \ge 0.5\\
0 & \text{if}~h_\theta(x) < 0.5
\end{cases}
$$

$g(z)$ 함수를 고려할 때, Dicision Boundary 다음과 같이 정리될 수 있다.

$$
y=
\begin{cases}
1 & \text{if}~\theta^Tx \ge 0\\
0 & \text{if}~\theta^Tx < 0
\end{cases}
$$

예를 들어 다음의 $h_\theta(x)$와 $\theta$가 결정되었다고 하자.

$$
h_\theta(x)=g\left(\theta_0+\theta_1x_1+\theta_2x_2\right),~\theta = [-3, 1, 1]
$$

이 경우 Dicision Boundary는 다음과 같이 정의된다.

$$
y=
\begin{cases}
1 & \text{if}~-3+x_1+x_2 \ge 0\\
0 & \text{if}~-3+x_1+x_2 < 0
\end{cases}
$$

<div style="text-align: center;">
    <img src="./Fig4.png" alt="nn" width="500">
</div><br>

## Learning Algorithms

다음으로 Logistic Regression의 Cost Function $J(\theta)$를 정의하고, gradient Desent로 이를 학습하는 방법에 대해 살펴보자.

시작에 앞서 Sigmoid Function의 유용한 미분 성질을 정리하자.

$$
\begin{align*}
g'(z)&=\frac{d}{dz}\frac{1}{1+e^{-z}}\\
&=\frac{1}{(1+e^{-z})^2}(e^{-z})\\
&=\frac{1}{(1+e^{-z})}\cdot \left(1-\frac{1}{(1+e^{-z})}\right)\\
&=g(z)(1-g(z))
\end{align*}
$$

이제 이진 분류문제의 확률적 모델링에서 $y$가 Bernoulli Distribution을 따른다고 가정하여 다음의 Probablistic assumption을 가정하자.

$$
\begin{align*}
p(y=1\mid x;\theta) &= h_\theta(x)\\
p(y=0\mid x;\theta) &= 1-h_\theta(x)
\end{align*}
$$

즉, 위의 가정을 종합하면 다음의 확률 밀도 함수를 정의할 수 있다.

$$
p(y\mid x;\theta) = (h_\theta(x))^y(1-h_\theta(x))^{1-y}
$$

이후 $m$개의 training sample에 대하여 independently and identically distributed, **IID**를 가정하여 다음의 likelihood를 얻을 수 있다.

$$
\begin{align*}
L(\theta) &= p(\vec{y}\mid X;\theta)\\
&=p(y^1\mid x^1;\theta)\times\cdots\times p(y^m\mid x^m;\theta)\\
&=\prod_{i=1}^mp(y^i\mid x^i;\theta)\\
&=\prod_{i=1}^m\left(h_\theta(x^i)\right)^{y^i}\left(1- h_\theta(x^i)\right)^{1-y^i}
\end{align*}
$$

또한 Linear Regression에서과 마찬가지로 log likelihood를 다음과 같이 정의한다.

$$
\begin{align*}
\ell(\theta) &= \log L(\theta)\\
&=\sum_{i=1}^my^i\log h_\theta(x^i)+(1-y^i)\log (1-h_\theta(x^i))
\end{align*}
$$

결론적으로 다음과 같이 cost function $J(\theta)$를 정의할 수 있다.

$$
J(\theta) = 
\begin{cases}
-\log h_\theta(x) & \text{if}~y=1\\
\log (1-h_\theta(x^i))& \text{if}~y=0
\end{cases}
$$

$y=1$인 경우는 아래와 같은 형태의 곡선으로 정의된다.

<div style="text-align: center;">
    <img src="./Fig5.png" alt="nn" width="400">
</div><br>

$y=0$인 경우는 아래와 같은 형태의 곡선으로 정의된다.

<div style="text-align: center;">
    <img src="./Fig6.png" alt="nn" width="400">
</div><br>

추가적으로 $\ell(\theta)$를 최대화하기 위해 gradient ascent 방법을 사용할 수 있다. (최소화를 위한 gradient descent와 달리 gradient ascent에서는 update formula에서 음의 부호 대신 양의 부호를 사용한다.)

$$
\theta := \theta+\alpha\nabla_{\theta}\ell(\theta)
$$

마지막으로 $\nabla_{\theta}\ell(\theta)$를 미분을 통해 구해보자.

$$
\begin{align*}
\frac{\partial}{\partial\theta_j}\ell(\theta) 
&=\left(y\frac{1}{g(\theta^Tx)}-(1-y)\frac{1}{1-g(\theta^Tx)} \right)\frac{\partial}{\partial\theta_j}g(\theta^Tx)\\
&=\left(y\frac{1}{g(\theta^Tx)}-(1-y)\frac{1}{1-g(\theta^Tx)} \right)g(\theta^Tx)(1-g(\theta^Tx))\frac{\partial}{\partial\theta_j}\theta^Tx\\
&=\left(y(1-g(\theta^Tx))-(1-y)g(\theta^Tx) \right)x_j\\
&=(y-h_\theta(x))x_j
\end{align*}
$$

최종적으로 다음의 stochastic gradient ascent rule를 얻을 수 있다.

$$
\theta_j:=\theta_j + \alpha\left(y^i-h_\theta(x^i)\right)x_j^i
$$

Linear Regression에서의 LMS(Least Mean Squares) 업데이트 규칙과 비교해보면, 로지스틱 회귀의 업데이트 규칙이 동일하게 보인다. 하지만, 이는 같은 알고리즘이 아니다. 왜냐하면, $h_\theta(x^i)$  가 이제는  $\theta^T x^i$  의 비선형 함수로 정의되기 때문이다.
그럼에도 불구하고, 서로 다른 학습 문제와 알고리즘에서 동일한 업데이트 규칙이 도출된다는 것은 흥미로운 일이다. 이것이 단순한 우연일까, 아니면 더 깊은 이유가 있을까? 이 질문에 대한 답은 일반화 선형 모델(GLM, Generalized Linear Models) 을 다룰 때 정리할 것이다.

### cf. Perceprton Algorithm

다음의 update rule은 Perceptron을 학습하는 것에서도 사용된다.

$$
\theta_j:=\theta_j + \alpha\left(y^i-h_\theta(x^i)\right)x_j^i
$$

하지만, 퍼셉트론이 로지스틱 회귀나 최소제곱법과 유사해 보일 수 있지만, 실제로는 완전히 다른 종류의 알고리즘이다. 특히, 퍼셉트론의 예측값은 확률적 해석이 어렵고, MLE로부터 유도할 수도 없다. 퍼셉트론에서는 오분류된 샘플이 있을 때만, 정의된 Loss에 따라 오분류를 수정하는 방향으로 학습하는 모델이라는 점에서 차이를 갖는다. 자세한 내용은 Chapter 3를 참고하자.

## Newton's Method

### Newton's Method

$\ell(\theta)$를 최대화 하는 다른 방법으로써 Newton's Method에 대해 정리하자. 

Newton's Method는 함수 $f(\theta):\mathbb{R}\rightarrow\mathbb{R}$에 대하여 $f(\theta)=0$인 $\theta \in \mathbb{R}$을 찾는 방법이다. 핵심 아이디어는 다음과 같다.

- 점 $(\theta_1, f(\theta_1))$에서 그은 접선의 $x$절편 $x=\theta_2$를 구한다.
- 점 $(\theta_2, f(\theta_2))$에서도 마찬가지의 방법을 반복한다.

수식으로 다시 정리하면 다음과 같다.

$$
\theta_n = \theta_{n-1} - \frac{f(\theta_{n-1})}{f'(\theta_{n-1})}
$$

이때, 이를 충분히 반복하여 다음을 구하여 $f(\theta)=0$인 $\theta \in \mathbb{R}$을 구할 수 있다.

$$
\lim_{n\rightarrow \infty}\theta_{n}
$$

기하학적인 관점에서 반복은 아래의 그림과 같이 수행된다.
<div style="text-align: center;">
    <img src="./Fig7.png" alt="nn" width="500">
</div><br>

이제 $\ell(\theta)$을 최대화하는 문제를, $\ell'(\theta)=0$을 찾는 문제로 이해하여 Newton's Method를 다음의 Update rule로 이해할 수 있다.

$$
\theta:=\theta-\frac{\ell'(\theta)}{\ell''(\theta)}
$$

### Newton-Raphson Method

다차원의 상황에서는 Hessian Matrix에 대해 다음과 같이 Update rule이 정의된다.

$$
\theta:=\theta-H^{-1}\nabla_\theta\ell(\theta)
$$

이때, $H$는 $n\times n$의 Hessian Matrix 이며 $\nabla_\theta\ell(\theta)$는 $\ell(\theta)$를 \theta에 대해 편미분한 Gradient vector이다.

정확하게 Hessian Matrix의 원소는 다음과 같이 정의된다.

$$
H_{ij}=\frac{\partial^2\ell(\theta)}{\partial\theta_i\partial\theta_j}
$$

참고로 이와 같이 Newton's Method를 Logistic Regression에서 $\ell(\theta)$를 최적화하는 데 적용되면 이를 Fisher Scoring이라고 부기도 한다.

### cf. Efficiency

- Newton's Method는 Gradient Descent보다 더 빠른 수렴 속도를 갖는다.
- Newton's Method는 더 적은 반복으로 최적해에 도달할 수 있다.
- $H$의 차원 수가 크면 $H^{-1}$를 계산하는 비용이 커진다.



### cf. Minimization

Maximization과 달리 Minimization은 Update 부호를 바꿈으로써 수행될 수 있다.

$$
\theta:=\theta+\frac{\ell'(\theta)}{\ell''(\theta)}
$$

