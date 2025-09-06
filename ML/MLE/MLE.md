# Maximum Likelihood Estimation (MLE)

## 일변량 가우시안 분포의 MLE

### Motivation
데이터셋 $\mathcal{X}=\{x_1, x_2, \cdots, x_n\}$ 이 정규분포 $\mathcal{N}(\mu, \sigma^2)$을 따를 것으로 추정하고, 이 데이터셋을 가장 잘 설명하는 parameter인 $\mu$와 $\sigma$를 추정하려고 한다.

### PDF
먼저 정규분포를 가정하였기에 $\mu, \sigma^2$에 대하여 데이터 $x$가 관측될 확률은 다음과 같다.

$$p(x|\mu, \sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

### IID
이때, 각각의 $x$가 독립적이고, 모든 $x$가 같은 분포에서 추출되었음을 가정하자. (independently and identically distributed, **IID**) 즉, $x$는 다음을 만족한다.

1. Independence (독립성)

$$
p(x_i, x_j) = p(x_i)\cdot p(x_j) \quad(i \not= j)
$$

2. Identically Distributed (동일분포)

$$
\forall x_i \in \mathcal{X} \sim p(x_i)
$$

### Likelihood

IID의 가정에 따라 Likelihood $L$을 다음과 같이 정의할 수 있다.

$$
L(\mu, \sigma^2) = \prod_{i=1}^{n}p(x_i\mid\mu,\sigma^2)
$$

Likelihood $L(\theta \mid x)$의 의미는 "어떤 데이터를 관측했을 때, 파라미터가 어떤 값일 가능성"을 의미하며, "어떤 파라미터가 주어졌을 때, 데이터가 나올 확률"인 $p(x|\theta)$와 수학적으로는 동일하지만 관점이 다르다. 즉 변수를 파라미터로 보느냐, 데이터로 보느냐의 차이를 나타낸다.

### Log-Likelihood

이제 우리는 Likelihood를 최대화하는 $\mu, \sigma^2$를 찾을 것이다. 즉, 이 데이터셋의 관측 확률이 최대가 되도록 하는 $\mu, \sigma^2$를 찾을 것이고, 이는 이 데이터셋을 가장 잘 설명하는 $\mu, \sigma^2$가 될 것이다. 따라서 계산상의 편의를 위해 Likelihood에 log를 취한다.

$$
\log L = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i-\mu)^2
$$

### MLE와 미분

이제 각각을 $\mu, \sigma^2$에 대해 미분하여 추정을 완성하자.

#### 1. $\mu$ 추정

$$
-\frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i-\mu) = 0
$$

$$
\therefore \hat{\mu} = \frac{1}{n}\sum_{i=1}^{n}x_i
$$

#### 2. $\sigma$ 추정

$$
-\frac{n}{\sigma} + \frac{1}{\sigma^3}\sum_{i=1}^{n}(x_i-\mu)^2=0
$$

$$
\therefore \hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i-\mu)^2=\mathbb{E}(X^2)-\mathbb{E}(X)^2
$$

## 다변량 가우시안 분포의 MLE

데이터 $\mathbf{x}_1, \cdots, \mathbf{x}_n \in \mathbb{R}^{d}$ 인 데이터셋 $\mathcal{X}$이 정규분포 $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$을 따를 것으로 추정하고, 이 데이터셋을 가장 잘 설명하는 parameter인 평균벡터 $\boldsymbol{\mu}$와 공분산 행렬 $\boldsymbol{\Sigma}$를 추정하려고 한다.

### PDF

$$
p(\mathbf{x}\mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{\frac{d}{2}}\sqrt{\det(\boldsymbol{\Sigma})}}\exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^{\top}\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)
$$

### Log-Likelihood

$$
\log L = -\frac{nd}{2}\log(2\pi)-\frac{n}{2}\log|\boldsymbol{\Sigma}|-\frac{1}{2}\sum_{i=1}^{n}(\mathbf{x}_i-\boldsymbol{\mu})^{\top}\boldsymbol{\Sigma}^{-1}(\mathbf{x}_i-\boldsymbol{\mu})
$$

### MLE와 미분 (row 기준 미분)

#### 1. $\boldsymbol{\mu}$ 추정

$$
\begin{align*}
\frac{\partial}{\partial \boldsymbol{\mu}} (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu}) \cdot \Delta\boldsymbol{\mu} &= (\mathbf{x}-\boldsymbol{\mu}-\Delta\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}-\Delta\boldsymbol{\mu}) -  (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\\
&= -(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}\Delta\boldsymbol{\mu}
-\underbrace{\Delta\boldsymbol{\mu}^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})}_{\text{scalar}} \\
&=-2(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}\Delta\boldsymbol{\mu}
\end{align*}
$$

$$
\begin{align*}
\therefore
\frac{\partial}{\partial \boldsymbol{\mu}} (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu}) =
-2(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}
\end{align*}
$$

임을 참고하여 계산하자.

$$
\frac{\partial \log L}{\partial \boldsymbol{\mu}} = \sum_{i=1}^{n} (\mathbf{x}_i - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} = 0
$$

$$
\therefore \hat{\boldsymbol{\mu}} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{x}_i
$$


#### 2. $\boldsymbol{\Sigma}$ 추정

1. $\frac{\partial}{\partial \boldsymbol{\Sigma}}\log |\boldsymbol{\Sigma}|$ 미분
$$
\begin{align*}
\frac{\partial}{\partial \boldsymbol{\Sigma}}\log |\boldsymbol{\Sigma}| \cdot \Delta\boldsymbol{\Sigma} &= \log(\det(\boldsymbol{\Sigma}+\Delta\boldsymbol{\Sigma})) - \log(\det(\boldsymbol{\Sigma})) \\
&= \log\left(\frac{\det(\boldsymbol{\Sigma})\cdot \det(\mathbf{I}+\boldsymbol{\Sigma}^{-1}\Delta\boldsymbol{\Sigma})}{\det(\boldsymbol{\Sigma})}\right)\\
&=\log(\det(\mathbf{I}+\boldsymbol{\Sigma}^{-1}\Delta\boldsymbol{\Sigma}))\\
&\approx\log(1+\text{tr}(\boldsymbol{\Sigma}^{-1}\Delta\boldsymbol{\Sigma}))\\
&\approx\text{tr}(\boldsymbol{\Sigma}^{-1}\Delta\boldsymbol{\Sigma})
\end{align*}
$$

$$
\therefore \frac{\partial}{\partial \boldsymbol{\Sigma}}\log |\boldsymbol{\Sigma}| = \boldsymbol{\Sigma}^{-1}
$$

2. $\frac{\partial}{\partial \mathbf{X}}  \mathbf{a}^\top \mathbf{X}^{-1} \mathbf{a}$ 미분
$$
\begin{align*}
\frac{\partial}{\partial \mathbf{X}}  \mathbf{a}^\top \mathbf{X}^{-1} \mathbf{a} \cdot\Delta\mathbf{X} &= \mathbf{a}^\top (\mathbf{X}+\Delta\mathbf{X})^{-1} \mathbf{a}-\mathbf{a}^\top \mathbf{X}^{-1} \mathbf{a} 
\end{align*}
$$

이때, $\mathbf{X}^{-1}=\mathbf{Y}$로 두고 다음의 수식을 통해 치환하자.

$$
(\mathbf{X}+\Delta\mathbf{X})(\mathbf{Y}+\Delta\mathbf{Y})=\mathbf{I}\\
\mathbf{I}+\mathbf{X}\Delta\mathbf{Y}+\Delta\mathbf{X}\mathbf{Y}+\Delta\mathbf{X}\Delta\mathbf{Y}=\mathbf{I}\\
\mathbf{X}\Delta\mathbf{Y}+\Delta\mathbf{X}\mathbf{Y}=0\\
\therefore \Delta\mathbf{Y}=-\mathbf{X}^{-1}\Delta\mathbf{X}\mathbf{X}^{-1}
$$

$$
\begin{align*}
\frac{\partial}{\partial \mathbf{X}}  \mathbf{a}^\top \mathbf{X}^{-1} \mathbf{a} \cdot\Delta\mathbf{X} &= \mathbf{a}^\top (\mathbf{X}+\Delta\mathbf{X})^{-1} \mathbf{a}-\mathbf{a}^\top \mathbf{X}^{-1} \mathbf{a} \\
&= \mathbf{a}^\top(\mathbf{Y}+\Delta\mathbf{Y})\mathbf{a}-\mathbf{a}^\top\mathbf{Y}\mathbf{a}\\
&= \mathbf{a}^\top\Delta\mathbf{Y}\mathbf{a}\\
&=-\mathbf{a}^\top\mathbf{X}^{-1}\Delta\mathbf{X}\mathbf{X}^{-1}\mathbf{a}\\
&=\text{tr}(-\mathbf{a}^\top\mathbf{X}^{-1}\Delta\mathbf{X}\mathbf{X}^{-1}\mathbf{a})\\
&=\text{tr}(-\mathbf{X}^{-1}\mathbf{a}\mathbf{a}^\top\mathbf{X}^{-1}\Delta\mathbf{X})
\end{align*}
$$

$$
\therefore \frac{\partial}{\partial \mathbf{X}}  \mathbf{a}^\top \mathbf{X}^{-1} \mathbf{a} = -\mathbf{X}^{-1}\mathbf{a}\mathbf{a}^\top\mathbf{X}^{-1}
$$

이제 위의 미분 공식을 이용하여 미분하자.

$$
\frac{\partial}{\partial \boldsymbol{\Sigma}} \left( -\frac{n}{2} \log |\boldsymbol{\Sigma}| \right) = -\frac{n}{2} \boldsymbol{\Sigma}^{-1}
$$

$$
\frac{\partial}{\partial \boldsymbol{\Sigma}} \left( -\frac{1}{2} \sum_{i=1}^{n} (\mathbf{x}_i - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x}_{i} - \boldsymbol{\mu}) \right)
= \frac{1}{2} \sum_{i=1}^{n} \boldsymbol{\Sigma}^{-1} (\mathbf{x}_i - \boldsymbol{\mu})(\mathbf{x}_i - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}
$$

$$
\therefore \frac{\partial \log L}{\partial \boldsymbol{\Sigma}} = -\frac{n}{2} \boldsymbol{\Sigma}^{-1} + \frac{1}{2} \sum_{i=1}^{n} \boldsymbol{\Sigma}^{-1} (\mathbf{x}_i - \boldsymbol{\mu})(\mathbf{x}_i - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} = 0
$$

$$
\therefore \hat{\boldsymbol{\Sigma}} = \frac{1}{n} \sum_{i=1}^{n} (\mathbf{x}_i - \hat{\boldsymbol{\mu}})(\mathbf{x}_i - \hat{\boldsymbol{\mu}})^\top
$$