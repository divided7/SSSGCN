# SSSGCN
It is currently being organized and will be open-sourced. The code and data will be made public after undergoing our de-identification review.
## Datasets
We collected real Tai Chi video data, which was professionally annotated with scores by sports experts. This data aims to explore potential complex action features, differing from traditional classification-based rating evaluations, such as grading actions as A, B, C, or D levels.

**Why do we use continuous variables as labels:** Although it may be cumbersome to modify the granularity of performance ratings established in classification tasks, it is generally possible to adjust them through methods such as reorganizing datasets and retraining models. Additionally, it is generally true that finer-grained classification tasks tend to be more challenging. Adopting smoothed labels and regression models can indeed lead to higher performance and finer-grained assessments, which better align with real examination and teaching scenarios. Although it requires more significant effort, this approach is more in line with real-world applications.

**Why don't We directly compare feature values as in facial recognition:** In action scoring tasks, directly comparing feature values may overlook spatial and temporal information of the actions. Additionally, sports experts have pointed out that the evaluation of scores should not solely rely on the similarity of actions; it involves a certain level of subjectivity or artistry. We aim for our data to provide this information and enable the model to represent it.

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
### NTU-RGB-D Ablation
<a href="https://colab.research.google.com/drive/1V0WdSHMwRdxWYtxeiRg-8-VZBceH6bdE?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> 

[Google Colab Demo](https://colab.research.google.com/drive/1V0WdSHMwRdxWYtxeiRg-8-VZBceH6bdE?usp=sharing) Note: the metrics in the Colab demo might experience slight variations due to version changes, but the overall performance should be approximately similar.

### Taichi Cls Ablation

## Reg Exp
### Taichi Scoring Reg Ablation
