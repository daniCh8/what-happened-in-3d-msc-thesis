# model backbones comparison
python train_classifier.py -en pointnet2 -mo m_sub -e 100 -b 4 -fp 0.2
python train_classifier.py -en kpconv -mo m_sub -e 100 -b 8 -fp 0.2
CUDA_VISIBLE_DEVICES=0 python train_classifier.py -en minkowski -mo m_sub -e 100 -b 128 -g 0 -fp 0.2

# 2d baseline backbones comparison
python train_classifier.py -en resnet18 -mo baseline -e 100 -b 8 -fp 0.2
python train_classifier.py -en resnet34 -mo baseline -e 100 -b 8 -fp 0.2
python train_classifier.py -en resnet50 -mo baseline -e 100 -b 8 -fp 0.2
python train_classifier.py -en resnext -mo baseline -e 100 -b 8 -fp 0.2

# multi view backbones comparison
python train_classifier.py -en resnet18 -mo multi_baseline -e 100 -b 4 -si 4 -fp 0.2
python train_classifier.py -en resnet34 -mo multi_baseline -e 100 -b 4 -si 4 -fp 0.2
python train_classifier.py -en resnet50 -mo multi_baseline -e 100 -b 2 -si 4 -fp 0.2
python train_classifier.py -en resnext -mo multi_baseline -e 100 -b 2 -si 4 -fp 0.2

# multi-view points of view comparison
python train_classifier.py -en resnet50 -mo baseline -e 100 -b 2 -si 1 -fp 0.2
python train_classifier.py -en resnet50 -mo multi_baseline -e 100 -b 2 -si 2 -fp 0.2
python train_classifier.py -en resnet50 -mo multi_baseline -e 100 -b 2 -si 3 -fp 0.2
python train_classifier.py -en resnet50 -mo multi_baseline -e 100 -b 2 -si 4 -fp 0.2
python train_classifier.py -en resnet50 -mo multi_baseline -e 100 -b 2 -si 5 -fp 0.2
python train_classifier.py -en resnet50 -mo multi_baseline -e 100 -b 2 -si 6 -fp 0.2

# data flip study
python train_classifier.py -en kpconv -mo m_sub -e 100 -b 8 -fp 0.0
python train_classifier.py -en kpconv -mo m_sub -e 100 -b 8 -fp 0.2
python train_classifier.py -en kpconv -mo m_sub -e 100 -b 8 -fp 0.4
python train_classifier.py -en kpconv -mo m_sub -e 100 -b 8 -fp 0.6
python train_classifier.py -en kpconv -mo m_sub -e 100 -b 8 -fp 0.8
python train_classifier.py -en kpconv -mo m_sub -e 100 -b 8 -fp 1.0
