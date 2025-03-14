# U-Net
U-Net: Convolutional Networks for Biomedical Image Segmentation [ 2015  ·  Olaf Ronneberger, Philipp Fischer, Thomas Brox ]

https://paperswithcode.com/paper/u-net-convolutional-networks-for-biomedical

## Abstract

- Biomedical Image segmentation을 위한 모델이고, FCN 기반의 Encoder-Decoder Based Model 이다.
- crop된 feature map을 직접적으로 전달하는 방식으로 convolution을 통해 손실된 가장자리 픽셀에 대한 정보 손실을 보완하고, 줄어든 feature map에 손실된 정보를 보완한다.
- 해당 환경에 적합한 여러 추가적 기법이 적용되었지만, 구조적인 측면에서 내용을 요약할 것이다.
    -  ex: Mirroring, Overlap-tile, data augmentation, elastic deformations, sliding window, loss function에 적용된 기법 등.

## Architecture

<div style="text-align: center;">
    <img src="./U-net.png" alt="nn" width="600">
</div><br>

- Encoder-Decoder의 대칭적인 구조를 갖는다.
- Contracting Path: 
    - 일반적인 CNN의 구조를 갖는다.
    - 각 size 별로 feature map을 crop하여 Expansive path에 전달한다.
- Expansive Path: 
    - $2\times2$ Up-conv를 통해 upsampling 한다.
    - Crop된 feature map과 upsampling된 feature map을 concat한다. (이를 통해 convolution을 통해 손실된 spatial information 및 border pixel에 대한 정보를 보존한다.)
- 단점: feature map을 crop하기 때문에 메모리 사용량이 높다.