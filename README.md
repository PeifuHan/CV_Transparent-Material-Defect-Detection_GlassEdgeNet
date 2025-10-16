# GlassEdgeNet: Multimodal Polarization-RGB Fusion with Dynamic Morphological Filtering for Transparent Material Edge Detection
# GlassEdgeNet: 多模态偏振-RGB融合与动态形态学滤波的透明材料边缘检测

[![PyTorch](https://img.shields.io/badge/PyTorch-1.0+-red.svg)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org)

## 📖 Overview | 概述

`GlassEdgeNet` is an advanced multimodal fusion network specifically designed to address the challenges of edge blurring and low contrast in defect detection of transparent materials (such as glass and resin). | `GlassEdgeNet` 是一个先进的多模态融合网络，专门用于解决透明材料（如玻璃、树脂等）在缺陷检测中的边缘模糊和低对比度难题。

This project innovatively integrates polarization imaging (AoLP/DoLP) with RGB information and combines curvature-adaptive dynamic morphological filtering, significantly improving the accuracy and robustness of transparent material edge detection. | 本项目创新性地融合偏振成像（AoLP/DoLP）与RGB信息，并结合曲率自适应的动态形态学滤波，显著提升了透明材料边缘检测的精度与鲁棒性。

### ✨ Core Features | 核心特性
- **Multimodal Feature Fusion**: Leverages high-frequency edge features from polarization information and low-frequency texture information from RGB data. | **多模态特征融合**：利用偏振信息的高频边缘特征与RGB数据的低频纹理信息。
  
- **Curvature Adaptive Filtering**: Dynamic morphological operations driven by local curvature estimation based on the Hessian matrix. | **曲率自适应滤波**：基于Hessian矩阵的局部曲率估计，驱动动态形态学操作。
  
- **Attention Mechanism**: Channel attention weighting fusion enhances feature representation. | **注意力机制**：通道注意力加权融合，增强特征表达能力。
  
- **End-to-End Training**: Complete PyTorch implementation supporting training and inference. | **端到端训练**：完整的PyTorch实现，支持训练与推理。

## 🏗 Model Architecture | 模型架构
### Key Techniques | 关键技术
1. **Multimodal Feature Extraction**
   - RGB branch: Extracts structural texture features.
   - Polarization branch: Fuses AoLP and DoLP to capture edge features.
   | **多模态特征提取**
   - RGB分支：提取结构纹理特征。
   - 偏振分支：融合AoLP和DoLP，捕获边缘特征。

2. **Frequency-Aware Fusion**
   - Discrete wavelet decomposition concept.
   - Polarization high-frequency + RGB low-frequency complementarity.
   | **频率感知融合**
   - 离散小波分解思想。
   - 偏振高频 + RGB低频互补。

## 📊 Dataset | 数据集
This project is based on the RGBP-Glass dataset, the first large-scale RGB-Polarization dataset for transparent material defect detection. | 本项目基于RGBP-Glass数据集，这是首个大规模面向透明材料缺陷检测的RGB-偏振数据集。

dataset/
├── train/
│   ├── image/        # RGB images (*_rgb.tiff)
│   ├── aolp/         # AoLP images (*_aolp.tiff)  
│   ├── dolp/         # DoLP images (*_dolp.tiff)
│   ├── mask/         # Mask labels (*_mask.png)
│   └── edge/         # Edge labels (*_edge.png)
└── test/
    └── ...           # Same structure
    
## 🚀 Quick Start | 快速开始
### Requirements | 环境要求
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install opencv-python pillow numpy scikit-learn
### Installation | 安装
git clone https://github.com/your-username/GlassEdgeNet.git
cd GlassEdgeNet
### Train the Model | 训练模型
python train.py --dataset_path /path/to/RGBP-Glass --epochs 50 --batch_size 4
### Inference and Prediction | 推理预测
python predict.py --model_path best_model.pth --input_dir test_images --output_dir results
### Configuration Parameters | 配置参数
DATASET_PATH = "RGBP-Glass/train"    # Dataset path | 数据集路径
MODEL_SAVE_PATH = "best_model.pth"   # Model save path | 模型保存路径
BATCH_SIZE = 2                       # Batch size | 批大小
LEARNING_RATE = 0.001               # Learning rate | 学习率
IMAGE_SIZE = (256, 256)             # Input size | 输入尺寸

## 📝 Citation | 引用
If you use this code or the RGBP-Glass dataset in your research, please cite the relevant literature: | 如果您在研究中使用了本代码或RGBP-Glass数据集，请引用相关文献：

@inproceedings{mei2022glass,
  title={Glass segmentation using intensity and spectral polarization cues},
  
  author={Mei, Haiyang and Dong, Bo and Dong, Wen and Yang, Jiaxi and Baek, Seung-Hwan and Heide, Felix and Peers, Pieter and Wei, Xiaopeng and Yang, Xin},
  
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  
  pages={12622--12631},
  
  year={2022}
  
}

## 🤝 Contributing | 贡献
Welcome to submit Issues and Pull Requests to improve this project! | 欢迎提交Issue和Pull Request来改进本项目！

### ⭐ If this project is helpful to you, please give us a Star! | ⭐ 如果这个项目对您有帮助，请给我们一个Star！
