# $\mathbf{S^3GCN}$: SPORT SCORING SIAMESE GRAPH CONVOLUTION NETWORK

It is currently being organized and will be open-sourced. The code and data will be made public after undergoing our de-identification review.

## News
We have implemented a scoring system for Nanquan(南拳) based on this projection, gif demo as:


## Datasets
<img src="https://github.com/user-attachments/assets/3b1fd9be-2068-4167-8845-5fff2fda48b5" alt="logo" width="30%" height="auto">

**The datasets and algorithm powerd by the Algorithm Department in [Hengonda](http://www.hengonda.com/)**

We collected real Tai Chi video data, which was professionally annotated with scores by sports experts. This data aims to explore potential complex action features, differing from traditional classification-based rating evaluations, such as grading actions as A, B, C, or D levels.

**Why do we use continuous variables as labels:** Although it may be cumbersome to modify the granularity of performance ratings established in classification tasks, it is generally possible to adjust them through methods such as reorganizing datasets and retraining models. Additionally, it is generally true that finer-grained classification tasks tend to be more challenging. Adopting smoothed labels and regression models can indeed lead to higher performance and finer-grained assessments, which better align with real examination and teaching scenarios. Although it requires more significant effort, this approach is more in line with real-world applications.

**Why don't We directly compare feature values as in facial recognition:** In action scoring tasks, directly comparing feature values may overlook spatial and temporal information of the actions. Additionally, sports experts have pointed out that the evaluation of scores should not solely rely on the similarity of actions; it involves a certain level of subjectivity or artistry. We aim for our data to provide this information and enable the model to represent it.

### Apply for datasets
**The data provided by us is only allowed for scientific purposes and commercial use is prohibited.**
You need to send a scanned PDF of the application to luyuxi@hengonda.com, include the following content:
* Region (country and city)
* Organization
* Research field
* Recent research by your team (about pose estimation)
* Team homepage
* Commitment and signature (handwriting)
### Augmentation

| ![8k_时间域增强分布](https://github.com/divided7/SSSGCN/assets/72434716/c78580fb-74aa-4c86-ab21-81d9792bee0f) | ![16k_时间域增强分布](https://github.com/divided7/SSSGCN/assets/72434716/c8d4de9e-7e56-4a20-9269-b7b4a12542e0) |
| :-------------------------------: | :---------------------------------: |
|        8k_aug      |         16k_aug         |

|  ![生成样本各分数段分布](https://github.com/divided7/SSSGCN/assets/72434716/223054b6-c7d0-4576-960c-81bef97c3894)   |  ![生成样本各分数段分布(principle=0 6)](https://github.com/divided7/SSSGCN/assets/72434716/d25775a2-38e8-4c92-9b66-9db78f84d94f)   |
| :--: | :--: |
|  principle=0.4   |  principle=0.6   |
|  ![生成样本各分数段分布(principle=1)](https://github.com/divided7/SSSGCN/assets/72434716/f98005b6-7714-4e0f-8416-14712b8ea2f3)   |  ![均衡后生成样本各分数段分布](https://github.com/divided7/SSSGCN/assets/72434716/94da7f39-e144-4c46-8167-3b8f1452d15b)   |
|  principle=1.0   |  clip   |


## One-Stage
We initially aimed to achieve both classification and regression simultaneously through a one-stage approach. However, despite our efforts, the final classification and regression performance (as shown in model iv) did not meet our expected metrics.Additionally, under the guidance of experts, we designed a reasonable data augmentation method.
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


i) Extract features using the ST-GCN backbone and feed the obtained feature map into both the classification and regression heads with CoLU.

ii) Building upon i, using the data augmentation.

iii) Building upon ii, split the feature map along the spatial dimension into two parts, and then separately feed them into the classification and regression heads.

iv) Building upon iii, concatenate the feature embeddings from the classification head with the input to the regression head.



## Two-Stage
## Cls Exp
![image](https://github.com/divided7/SSSGCN/assets/72434716/7123c397-661c-4718-8dfc-42aa3e5b70d1)
![image](https://github.com/divided7/SSSGCN/assets/72434716/60c8594d-4fef-49a2-9812-a5c9052a20d5)
![image](https://github.com/divided7/SSSGCN/assets/72434716/9d9b81f8-af7d-4afb-a324-12731360f32e)
![output_SkipFrame=10_WinSize=355](https://github.com/user-attachments/assets/79da7c4e-83f7-4c30-973d-d36718ef8e8e)

### NTU-RGB-D Ablation
ST-GCN vs STD-GCN vs SST-GCN vs SSTD-GCN vs ST-GCN++ vs SSTD-GCN++ Demo

<a href="https://colab.research.google.com/drive/1V0WdSHMwRdxWYtxeiRg-8-VZBceH6bdE?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> 

[Google Colab Demo](https://colab.research.google.com/drive/1V0WdSHMwRdxWYtxeiRg-8-VZBceH6bdE?usp=sharing) Note: the metrics in the Colab demo might experience slight variations due to version changes, but the overall performance should be approximately similar.

### Taichi Cls Ablation
<a href="https://colab.research.google.com/drive/1qRGd1qwgZ8h9MNg3TCbSpOy7b50atl59?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> 

[Google Colab Demo](https://colab.research.google.com/drive/1qRGd1qwgZ8h9MNg3TCbSpOy7b50atl59?usp=sharing)

| Model         | NTU-RGB-D | Taichi   | Param. (M) | FLOPs (G) |
|---------------|------------|----------|------------|-----------|
| ST-GCN        | 76.00%     | 65.47%   | 0.17       | 0.20      |
| STGL-GCN             | 77.50%     | 83.75%   | 2.78       | 1.89      |
| **SSTD-GCN(ours)**        | **87.00%** | **99.17%** | **0.18** | **0.11** |
| ST-GCN++          | 90.50%     | 93.33%   | 3.09       | 0.60      |
| SSTD-GCN++(ours is embedded to ST-GCN++)      | **92.00%** | **99.58%** | **0.32** | **0.61** |

## Reg Exp
### Taichi Scoring Reg Ablation

| Model        | Spacial Separate | Temporal Dilation | Taichi score MAE |
|--------------|------------------|-------------------|------------------|
| ix           | ❌               | ❌                | 0.0355           |
| x            | ❌               | ❌                | 0.0295           |
| xi           | ❌               | ✔️                | 0.0243           |
| xii          | ✔️               | ❌                | 0.0261           |
| **xiii**     | **✔️**           | **✔️**            | **0.0196**       |

