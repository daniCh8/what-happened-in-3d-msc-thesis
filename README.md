# What Happened in 3D, 2022, Master's Thesis

<!-- TOC -->

-   [What Happened in 3D, 2022, Master's Thesis](#what-happened-in-3d-2022-masters-thesis)
    -   [Abstract](#abstract)
    -   [Model](#model)
        -   [Three-Dimensional Approach](#three-dimensional-approach)
        -   [Two-Dimensional Baseline](#two-dimensional-baseline)
        -   [Multi-View Baseline](#multi-view-baseline)
    -   [Requirements](#requirements)
        -   [Dataset](#dataset)
        -   [Environment](#environment)
        -   [RScan](#rscan)
    -   [Run Training](#run-training)
        -   [Model Selection](#model-selection)
        -   [Data Augmentation](#data-augmentation)
    -   [Check Model Performance](#check-model-performance)
        -   [Metrics Generation](#metrics-generation)
            -   [Sample Metrics Output: Confusion Matrix](#sample-metrics-output-confusion-matrix)
            -   [Sample Metrics Output: tSNE Plot](#sample-metrics-output-tsne-plot)
        -   [Inference](#inference)
    -   [Pre-Trained Models](#pre-trained-models)
        -   [K3D](#k3d)
        -   [ResNet50-Baseline](#resnet50-baseline)
        -   [MultiView-Baseline](#multiview-baseline)

<!-- /TOC -->

## Abstract

Interpreting three-dimensional scenes that change over time is an open topic at its early stages. Current research mainly focuses on marking the existence of a difference between two temporal instances of the same scene without specifying the semantics of such difference. However, providing information on what has happened in an environment that changed over time can be helpful to a user only if the generated data is informative on the changes.

In this project, I introduce _Indoor Scene Activity Recognition_, a new deep learning challenge that aligns elements of Action Classification and Change Detection. I annotate and collect a new dataset for this task by analyzing and manipulating another publicly available dataset, [3RScan](https://waldjohannau.github.io/RIO/), and I develop K3D, a dedicated two-stream three-dimensional neural network to tackle the challenge. Additionally, I create a two-dimensional convolutional baseline and an object multi-view baseline to benchmark K3D and compare the results in different settings.

## Model

### Three-Dimensional Approach

To succeed in the Indoor Action Recognition task, the neural network must understand and focus on the scene difference. I aimed to build an architecture able to both interpret the input pointclouds separately and compare them to find what has changed.

For this reason, I created a two-stream architecture: the first stream processes the input features separately, and the second stream mixes them together to provide a comparison. The figure below shows the neural network.

![sub-model](/assets/sub-model.png)

I will introduce each building block separately:

-   **Inputs**: objects pointclouds.
-   **Feature Extractor**: the architecture used to process the pointclouds. Three different point convolutions are available: KPConv, Minkowski, and PointCloud++.
-   **Double Stream**: the outputs of the feature extractor block are two feature vectors representing the reference object and the changed object, respectively. Two architectural branches follow the feature extractor. In the first branch, the two feature vectors are concatenated and processed through a linear layer. In the second branch, the two feature vectors are subtracted and joined to the output of the first branch.
-   **Dense Sequence**: the sequence of linear layers used to process the double stream block's output.
-   **Action Prediction**: The final classification action prediction.

### Two-Dimensional Baseline

The two-dimensional approach is the first and most simple baseline I will define. I used a similar architecture in both approaches. I simplified the two-stream architecture to a single-stream and replaced the point convolutions with two-dimensional feature extractors. Below is a figure showing the architecture.

![2d-model](/assets/2d-model.png)

Let's go over each block:

-   **Inputs**: rendered pictures before and after the change, and bounding boxes.
-   **Feature Extractor**: the architecture used to process the images. The available ones are ResNet18, ResNet34, ResNet50, and ResNeXt.
-   **Dense Sequence**: the sequence of linear layers used to process the feature extractor's concatenated outputs.
-   **Action Prediction**: the final classification action prediction.

### Multi-View Baseline

The two-dimensional baseline works well, but it's also at a substantial disadvantage to our architecture since it doesn't have any three-dimensional information to work with. To better benchmark our framework, I defined a multi-view baseline that encodes the 3D knowledge of the scene into projected pictures from different viewpoints in the scan.

To build the model for the multi-view scenario, I built up from the single viewpoint network. The main differences between the two approaches are the feature extractor's output size and the concatenation of the feature vectors. The former had to be reduced for the number of parameters not to explode when using numerous viewpoints. The latter had to be adapted to pair the feature vectors of each viewpoint pair between each other. The figure below shows the approach.

![multiview-model](/assets/multiview-model.png)

Let's go over the building blocks:

-   **Inputs**: rendered multi-view pictures before and after the change, and bounding boxes.
-   **Feature Extractor**: the architecture used to process the images.
-   **Dense Sequence**: the sequence of linear layers used to process the feature extractor's concatenated outputs. Each viewpoint is paired before being joined with the others.
-   **Action Prediction**: the final classification action prediction.

## Requirements

### Dataset

The dataset can be found [at this link](https://1drv.ms/u/s!Am0bH-nj75Y6x4FGH8CSxh1sEBOFBA?e=akxH1X). The tar file should be downloaded and extracted in `/indoor-scene-activity-recognition/`. The structure of the dataset is the following:

```
/indoor-scene-activity-recognition/
|-- <scan_id>
    |-- caption.txt
    List of changes in this <scan_id>
    |-- meta.txt
    Metadata for this <scan_id>
    <action_dir>
    |-- baseline/
        |-- ref
            |-- img_i.jpg
            i rendered image of the before-action object
            |-- bb_i.txt
            i bounding box of the before-action object in img_i.jpg
        |-- chg
            |-- img_i.jpg
            i rendered image of the after-action object
            |-- bb_i.txt
            i bounding box of the after-action object in img_i.jpg
    |-- isolated_obj/
        |-- ref
            |-- mesh.obj
            Isolated before-action object mesh
            |-- mesh.mtl
            Corresponding material file
            |-- mesh_0.png
            Corresponding mesh texture
        |-- chg
            |-- mesh.obj
            Isolated after-action object mesh
            |-- mesh.mtl
            Corresponding material file
            |-- mesh_0.png
            Corresponding mesh texture
    |-- isolated_ply/
        |-- ref
            |-- new_ref_located_ascii.ply
            ASCII semantic segmentation information for before-action object
            |-- new_ref_located_pyntcloud.ply
            Binary encoded semantic segmentation information for before-action object
            |-- new_ref_located_ascii.ply
            CSV semantic segmentation information for before-action object
        |-- chg
            |-- new_chg_located_ascii.ply
            ASCII semantic segmentation information for after-action object
            |-- new_chg_located_pyntcloud.ply
            Binary encoded semantic segmentation information for after-action object
            |-- new_chg_located_ascii.ply
            CSV semantic segmentation information for after-action object
    |-- masks/
        |-- mask_ref_20000.npy
        Before-action object mask in pointcloud
        |-- mask_chg_20000.npy
        After-action object mask in pointcloud
    |-- npy/
        |-- 20000
            |-- ref_20000.npy
            Before-action object pointcloud
            |-- chg_20000.npy
            After-action object pointcloud
```

### Environment

The conda environment is specified in the [environment.yml](/environment.yml) file. A new environment should be created from it:

```
conda env create -f environment.yml
```

The new created environment's name will be `k3d`.

### 3RScan

The Indoor Scene Activity Recognition dataset has been built on top of the 3RScan dataset. The latter is not required to use the former, and the repository is already set up not to use the 3RScan information. However, if you want to dig deeper into this project, and unlock the full potential of the code, you can download the 3RScan dataset and extract it in `/3RScan/data/`. More information on how to download 3RScan can be found on the [project's official repository](https://github.com/WaldJohannaU/3RScan).

## Run Training

To run the training, let's first of all activate the right conda environment, and change directory to the source folder.

```bash
conda activate k3d
cd /src
```

We can then run the training using the following script:

```bash
python train_classifier.py [-d --data-path] [-cd --captions-path] [-lr --learning-rate] [-b --batch-size] [-e --epochs] [-m --metrics] [-lr_dec --learning-rate-decay] [-s --sample-size] [-ed --embed-dim-detector] [-q --quantization-size] [-en --encoder] [-ms --manual-seed] [-c --cpu] [-g --gpu] [-mo --model] [-si --siamese] [-fp --flip-prob]
```

Let's break down the argument of the call:

-   `-d --data-path` is the path to 3RScan. The default is set to `None`, and there's no need to change it unless to extend the dataset.
-   `-cd --captions-path` is the path to the Indoor Scene Activity Recognition dataset. Default is `'../indoor-scene-activity-recognition/'`.
-   `-lr --learning-rate` is the learning rate. Default is `.001`.
-   `-b --batch-size` is the batch size. Default is `16`.
-   `-e --epochs` is the number of epochs. Default is `100`.
-   `-m --metrics` controls how often the metrics are computed. Default is `5`.
-   `-lr_dec --learning-rate-decay` is the learning rate decay. Default is `.5`.
-   `-s --sample-size` is the input pointcloud dimension. Default is `20000`.
-   `-ed --embed-dim-detector` is the dimension of the change detector. Default is `128`.
-   `-q --quantization-size` is the quantization size for the minkowski engine input generation. Default is `.1`.
-   `-en --encoder` is the encoder type used. Choices are `['kpconv', 'pointnet2', 'minkowski', 'resnet18', 'resnet34', 'resnet50', 'resnext']`. Default is KPConv.
-   `-ms --manual-seed` is the random seed. Default is `0`.
-   `-c --cpu` controls if to run the training on CPU. Default is `False`.
-   `-g --gpu` controls the index of the GPU to use. Default is `-1` (best available GPU will be used).
-   `-mo --model` controls the type of model used. Choices are `['full_features', 'full_attention', 'm_features', 'm_attention', 'm_ensemble', 'm_sub', 'nm_features', 'nm_attention', 'baseline', 'multi_baseline']`.
-   `-si --siamese` controls the number of baseline input pictures to use. Default is `1`.
-   `-fp --flip-prob` controls the augmentation probability. Default is `.0`.

The default parameter run the training for the best performing model in the research.

### Model Selection

The available models/encoders pairs are the following:

-   The [three-dimensional approach](#three-dimensional-approach) has three backbone options available.
    ```
    model='m_sub'
    encoder=['kpconv','minkowski','pointnet2']
    ```
-   The [two-dimensional baseline](#two-dimensional-baseline) has four backbone options available.
    ```
    model='baseline'
    encoder=['resnet18','resnet34','resnet50','resnext']
    ```
-   The [multi-view baseline](#multi-view-baseline) has four backbone options available, and five options of input images available.
    ```
    model='multi_baseline'
    encoder=['resnet18','resnet34','resnet50','resnext']
    si=[2,3,4,5,6]
    ```

### Data Augmentation

I structured the dataset so that each category has a unique opposite class. This class pairing allows to easily augment the data by doing random input flipping on the data samples. In theory, the size of the dataset could be doubled by feeding, for each data sample, both the original input and the flipped one. However, surprisingly, experimental results show that flipping does not end up helping the performance of the model. To use data flipping, set the `fp` parameter of the training script.

## Check Model Performance

### Metrics Generation

To investigate on the performances of the models, I created a script that generates a classification score, a classification heatmap, and a t-SNE plot. It can be used in the following way:

```bash
python classifier_metrics.py [-ck --ckpt] [-c --cpu] [-g --gpu]
```

Let's break down the argument of the script:

-   `-ck --ckpt` sets the path to the trained folder.
-   `-c --cpu` controls if to run the training on CPU. Default is `False`.
-   `-g --gpu` controls the index of the GPU to use. Default is `-1` (best available GPU will be used).

#### Sample Metrics Output: Confusion Matrix

Below is a sample confusion matrix outputted by the script...

![confusion-matrix](/assets/training_confusion_matrix.png)

#### Sample Metrics Output: tSNE Plot

...and a sample tSNE plot outputted by the script:

![confusion-matrix](/assets/training_tsne.png)

### Inference

To check on individual samples, I created another script that can be used in the following way:

```bash
python classifier_inference.py [-ck --ckpt] [-c --cpu] [-g --gpu] [-k --key] [-chg --changed]
```

-   `-ck --ckpt` sets the path to the trained folder.
-   `-c --cpu` controls if to run the training on CPU. Default is `False`.
-   `-g --gpu` controls the index of the GPU to use. Default is `-1` (best available GPU will be used).
-   `-k --key` can be used paired to `-chg --changed to set a specific inference case.
-   `-chg --changed` can be used paired to `-k --key` to set a specific inference case.

Below is a sample inference output.

```bash
569d8f1c-72aa-2f24-89bb-0df7a0653c26/12/shifted:
        PREDICTED: ['tidied up']; GROUNDTRUTH: ['shifted']

531cfefe-0021-28f6-8c6c-35ae26d2158f/18/moved:
        PREDICTED: ['moved']; GROUNDTRUTH: ['moved']

8e0f1c2f-9e28-2339-85ae-05fc50d1a3a7/41/added:
        PREDICTED: ['added']; GROUNDTRUTH: ['added']

75c259a5-9ca2-2844-9441-d72912c1e696/109/added:
        PREDICTED: ['added']; GROUNDTRUTH: ['added']

20c993bf-698f-29c5-8549-a69fd169c1e1/81/added:
        PREDICTED: ['added']; GROUNDTRUTH: ['added']

531cff10-0021-28f6-8f94-80db8fdbbbee/25/rotated:
        PREDICTED: ['tidied up']; GROUNDTRUTH: ['rotated']

20c9939b-698f-29c5-85a4-68c286bd7053/8/shifted:
        PREDICTED: ['shifted']; GROUNDTRUTH: ['shifted']

8e0f1c2f-9e28-2339-85ae-05fc50d1a3a7/44/added:
        PREDICTED: ['added']; GROUNDTRUTH: ['added']

8eabc41a-5af7-2f32-8677-c1e3f9b04e62/30/shifted:
        PREDICTED: ['tidied up']; GROUNDTRUTH: ['shifted']

8eabc41a-5af7-2f32-8677-c1e3f9b04e62/6/rearranged:
        PREDICTED: ['shifted']; GROUNDTRUTH: ['rearranged']

7ab2a9c5-ebc6-2056-89c7-920e98f0cf5a/44/removed:
        PREDICTED: ['removed']; GROUNDTRUTH: ['removed']

10b17967-3938-2467-88c5-a299519f9ad7/3/rearranged:
        PREDICTED: ['closed']; GROUNDTRUTH: ['rearranged']

10b1792a-3938-2467-8b4e-a93da27a0985/36/tidied up:
        PREDICTED: ['tidied up']; GROUNDTRUTH: ['tidied up']

751a557f-fe61-2c3b-8f60-a1ba913060c4/34/shifted:
        PREDICTED: ['tidied up']; GROUNDTRUTH: ['shifted']

10b1792e-3938-2467-8bb3-172148ae5a67/14/shifted:
        PREDICTED: ['shifted']; GROUNDTRUTH: ['shifted']

8eabc418-5af7-2f32-85a1-a2709b29c46d/7/removed:
        PREDICTED: ['added']; GROUNDTRUTH: ['removed']
```

## Pre-Trained Models

### K3D

The best performing model of the research. It's from the [Three-Dimensional Approach](#three-dimensional-approach) approach, and uses a KPConv backbone. It can be downloaded [at this link](https://1drv.ms/u/s!Am0bH-nj75Y6x4FHrETV4EviOwoBoA?e=htEdjo).

### ResNet50-Baseline

The best performing two-dimensional baseline. It's from the [Two-Dimensional Baseline](#two-dimensional-baseline) approach, and uses a ResNet50 backbone. It can be downloaded [at this link](https://1drv.ms/u/s!Am0bH-nj75Y6x4FKsMT3kF8D_m8-kw?e=Ls6rOO).

### MultiView-Baseline

The best performing model of the research. It's from the [Multi-View Baseline](#multi-view-baseline) approach, and uses a ResNet50 backbone and 6 different input points of view. It can be downloaded [at this link](https://1drv.ms/u/s!Am0bH-nj75Y6x4FJb3HdCaqYFS1hBQ?e=cAzSD7).
