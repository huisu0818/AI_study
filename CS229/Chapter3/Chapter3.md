# CS229: Chapter3

- Generalized Linear Models (GLM)
- Lecture 4

## Study Members
- Huisu Lee (Lecture Note Summary)
    - Ajou University, Software and Computer Engineering
    - Computer Vision, Deep Learning

- Dogyu Lee (Code Implementation)
    - Seoul National University, Electrical and Computer Engineering
    - Deep Learning Hardware Acceleration, Optimization

## Contents
- Perceptron
- The exponential family
- Constructing GLMs
- Ordinary Least Squares
- Logistic Regression
- Softmax Regression

## Perceptron


$$
p(y; \eta) = b(y) \exp(\eta^T T(y) - a(\eta))
$$

$$
p(y; \mu) = \frac{1}{\sqrt{2 \pi}} \exp\left(-\frac{1}{2}(y - \mu)^2\right) = \frac{1}{\sqrt{2 \pi}} \exp\left(-\frac{1}{2}y^2\right) \cdot \exp\left(\mu y - \frac{1}{2}\mu^2\right)
$$

$$
\frac{\partial}{\partial \theta_j} J(\theta) = \frac{\partial}{\partial \theta_j} \frac{1}{2} (h_\theta(x) - y)^2 = 2 \cdot \frac{1}{2} (h_\theta(x) - y) \cdot \frac{\partial}{\partial \theta_j} (h_\theta(x) - y) = (h_\theta(x) - y) \cdot \frac{\partial}{\partial \theta_j} \left( \sum_{i=0}^{n} \theta_i x_i - y \right) = (h_\theta(x) - y) x_j
$$