# SSSGCN
It is currently being organized and will be open-sourced. The code and data will be made public after undergoing our de-identification review.
## One-Stage
We initially aimed to achieve both classification and regression simultaneously through a one-stage approach. However, despite our efforts, the final classification and regression performance (as shown in model iv) did not meet our expected metrics.
### Model Structure

i) and ii)

<img src="https://github.com/divided7/SSSGCN/assets/72434716/64949cdd-0f70-4f16-8e95-314db763bc70" alt="image" width="700"/>

iii)

<img src="https://github.com/divided7/SSSGCN/assets/72434716/83d9fd28-a2b0-45bf-9b24-58e1157e6afe" alt="image" width="700"/>

iv)

<img src="https://github.com/divided7/SSSGCN/assets/72434716/25db9e10-eaed-4869-91f2-4db914effa0c" alt="image" width="700"/>

### Exp

| Model | Taichi score MAE | Taichi classification Acc |
|:-----:|:----------------:|:-------------------------:|
|   i   |      0.2021      |          59.17%           |
|  ii   |      0.0965      |          84.42%           |
|  iii  |      0.0862      |          86.26%           |
|  iv   |      0.0782      |          95.58%           |





## Two-Stage
## Cls Exp
### NTU-RGB-D Ablation
<a href="https://colab.research.google.com/drive/1V0WdSHMwRdxWYtxeiRg-8-VZBceH6bdE?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> 

[Google Colab Demo](https://colab.research.google.com/drive/1V0WdSHMwRdxWYtxeiRg-8-VZBceH6bdE?usp=sharing) Note: the metrics in the Colab demo might experience slight variations due to version changes, but the overall performance should be approximately similar.

### Taichi Cls Ablation

## Reg Exp
### Taichi Scoring Reg Ablation
