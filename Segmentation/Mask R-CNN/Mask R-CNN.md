# Mask R-CNN
Mask R-CNN [ ICCV 2017  ·  Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick ]

- 논문링크: https://paperswithcode.com/paper/mask-r-cnn
- 참고자료1: https://deep-math.tistory.com/26
- 참고자료2: https://herbwood.tistory.com/20

## Abstract

- Faster R-CNN에 Mask branch 구조를 추가하여 object detection 모델을 instance segmentation 모델로 전환한 것
- 각 Faster R-CNN의 RPN(Region propsal Network)에서 얻은 RoI(Region of Interest)에 FCN을 적용하여 segmentation을 구현한 것이라고 이해할 수 있다.

## Faster R-CNN

<div style="text-align: center;">
    <img src="./Mask R-CNN7.png" alt="nn" width="500">
</div><br>

<div style="text-align: center;">
    <img src="./Mask R-CNN1.png" alt="nn" width="500">
</div><br>

- Region Proposal Network(RPN)은 Convolution을 통해 생성된 feature map을 입력으로 받아서 객체에 대한 Score와 함께 사각형의 물체 Proposal을 추출한다. 
- Fast R-CNN과 계산을 공유하기 위해 Fully Convolution으로 구성.
- Region Proposal을 생성하기 위해 feature map을 입력으로 받아 사용하는 Convolution Network를 구성하고, Sliding Window를 통해 Region of Interest를 생성 (RoI)
- 이후 RoI와 기존 feature map을 RoI pooling하여 고정된 크기의 feature map을 얻고, 이를 통해 Box-regression과 classification을 수행한다.

## Anchors

<div style="text-align: center;">
    <img src="./Mask R-CNN2.png" alt="nn" width="500">
</div><br>

- 먼저 원본 이미지에서 feature map이 생성될 때, 그 비율을 기준으로 원본 이미지의 영역을 grid로 나눠줄 수 있다. 이때, anchor box가 이 grid의 중심을 기준으로 각각의 grid cell에 대해 생성된다.
- Anchor box는 9개가 정의되어 있는데, 각각 서로 다른 size와 ratio를 갖는다. 이를 통해 다양한 크기의 객체를 인식할 수 있게 된다.
- RPN에서 anchor를 통해 $9WH\over{\text{Down Sampling ratio}}$ 의 Region Proposal이 생성된다. RPN은 이 각각의 Region Proposal에 대해 Score를 측정하는데, 여기서 Score는 Region Proposal에 객체가 포함되는지 아닌지를 의미하고, 각 anchor 마다 score를 갖고 있게 된다. 즉, feature map에 $1\times 1$ conv를 적용하여 channel 수가 $2\times 9$가 되도록 설정하여 새로운 feature map을 생성한다.
- 다음으로 box regressor를 얻기 위해 위와 마찬가지로 $1\times 1$ conv를 적용하여 channel 수가 $4\times 9$가 되도록 설정하여 새로운 feature map을 생성한다. ($\text{Box}_{regressor} = (x, y, w, h)$ 여야 하기에 4)
- 최종적으로 생성된 Region Proposal에 대해 상위 N개만을 score에 따라 추출하고, 기존 feature map과 함께 RoI pooling하여 classifier와 box regressor에 전달한다.
- 결과적으로 RPN은 유용한 RoI를 추출하기 위한 것이고, 세부적인 class prediction과 box regression은 이후의 classifier와 box regressor에서 수행된다.


<div style="text-align: center;">
    <img src="./Mask R-CNN3.png" alt="nn" width="500">
</div><br>

- 객체의 위치를 인지하기 위해 기존에 사용되는 방법은 크게 두 가지가 있다.
    - 첫 번째는 Feature Pyramid 방식인데, 이는 먼저 영상을 Resize하여 여러 size의 영상으로 복제하고, 각각에 대해 feature를 뽑아낸다. 이는 연산량이 많아서 시간이 오래걸리게 되는 단점을 갖고 있다.
    - 두 번째 방법은 여러 크기의 Sliding Window를 쓰는 것이다.
- 마지막으로 해당 논문에서 사용된 pyramid of anchor 방식은. 계산량도 적고 사진도 한 장에 대해서만 계산하면 된다. 이 방법은 여러 크기와 비율의 객체를 빠르게 찾을 수 있다는 장점이 있다.


## Architecture

<div style="text-align: center;">
    <img src="./Mask R-CNN4.png" alt="nn" width="500">
</div><br>

Mask R-CNN은 Faster R-CNN의 RPN(Region Proposal Network)에서 얻은 RoI(Region of Interest)에 대하여 객체의 class를 예측하는 classification branch, box regression을 수행하는 box regression branch와 평행하게 segmentation mask를 예측하는 mask branch를 추가한 구조를 가지고 있다. mask branch는 RPN에서 생성된 각각의 추출된 RoI에 대해 FCN(Fully Convolutional Network)를 추가한 형태이다. segmentation task를 보다 효과적으로 수행하기 위해서는 spatial information을 보존해야 하는데, 기존의 RoI Pooling은 이를 보존하지 못한다는 문제점을 갖고 있었다. 따라서 객체의 spatial location을 보존할 수 있는 RoI Align 기법을 통해서 segmentation을 수행할 수 있도록 모델을 수정하였다.

<div style="text-align: center;">
    <img src="./Mask R-CNN5.png" alt="nn" width="500">
</div><br>

- RoI Pooling:
    - RoI Pooling은 각 RoI에서 small feature map을 추출하는 연산이다.
    - 다른 사이즈의 Region Proposal을 입력으로 받더라도, max pooling을 이용하여 output size를 동일하게 만들어 feature map을 생성한다.
    - 동일한 size로 만드는 과정에서 RoI와 추출된 feature 사이에 matching이 제대로 이뤄지지 않기 (오정렬 발생) 때문에 픽셀 단위로 예측하는 segmentation mask에는 단점이 된다.


<div style="text-align: center;">
    <img src="./Mask R-CNN6.png" alt="nn" width="500">
</div><br>

- RoIAlign:
    - Bilinear interpolation 연산을 사용하여 각 RoI bin의 샘플링된 4개의 위치에서 input feature의 정확한 값을 계산 하여 결과를 max 혹은 avg 처리한다.