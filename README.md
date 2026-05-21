# 🚲 基于YOLO26n的共享单车检测系统（香橙派AIpro部署）

本项目基于YOLO26n目标检测算法，实现城市共享单车停放区域的智能检测，可识别共享单车位置与数量，支持PyTorch训练、ONNX导出与香橙派AIpro边缘端部署，为城市单车管理提供轻量化、低功耗的AI解决方案。

---

## 📋 目录
- [项目背景](#项目背景)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [环境依赖](#环境依赖)
- [快速开始](#快速开始)
  - [模型训练](#模型训练)
  - [本地PyTorch推理](#本地PyTorch推理)
  - [ONNX模型导出与测试](#ONNX模型导出与测试)
- [香橙派AIpro部署指南](#香橙派AIpro部署指南)
- [训练结果与评估](#训练结果与评估)
- [注意事项](#注意事项)
- [许可证](#许可证)

---

## 🌆 项目背景
随着城市共享单车数量激增，乱停乱放问题给城市管理带来挑战。传统人工巡查效率低、成本高，本项目通过边缘AI设备实现实时单车检测，为城市精细化管理提供低成本、易部署的技术方案。

---

## 🛠️ 技术栈
| 模块 | 技术/工具 |
|------|-----------|
| 基础框架 | Python 3.8+ |
| 目标检测 | YOLO26n (Ultralytics) |
| 模型格式 | PyTorch (.pt) / ONNX / 昇腾.om |
| 开发板 | 香橙派 AIpro (昇腾310芯片) |
| 部署工具 | ONNX Runtime / MindX SDK |
| 数据标注 | LabelImg / LabelStudio |

---

## 📁 项目结构
```text
cycledata/
├── configs/
│   └── shared_bicycle.yaml   # 数据集路径、类别与训练超参数配置
├── dataset/
│   ├── images/               # 训练/验证/测试图片（未提交）
│   └── labels/               # 对应标注文件（YOLO格式，未提交）
├── models/
│   └── weights/              # 预训练权重存放目录（未提交）
├── onnx_convert/             # ONNX格式转换相关工具
├── scripts/
│   ├── runs/
│   │   └── detect/train-2/   # 训练日志与结果
│   │       ├── weights/
│   │       │   ├── best.pt   # 最优PyTorch权重
│   │       │   └── best.onnx # 导出的ONNX模型
│   │       ├── confusion_matrix.png  # 混淆矩阵
│   │       ├── BoxPR_curve.png       # PR曲线
│   │       └── results.csv           # 训练过程指标
│   ├── export_onnx.py       # PyTorch模型转ONNX脚本
│   ├── test_pt.py            # PyTorch模型本地推理测试
│   ├── test_onnx_model.py    # ONNX模型推理测试
│   └── train.py              # 模型训练主脚本
├── test.jpg                  # 测试用示例图片
├── yolo26n.pt                # YOLO26n预训练权重
├── .gitignore                # Git忽略文件配置
└── README.md                 # 项目说明文档

# 安装基础依赖
pip install ultralytics opencv-python numpy matplotlib onnxruntime
# 如需ONNX转换优化
pip install onnx onnxsim

# 数据集结构
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

## 香橙派部署指南
### 模型转换：使用atc工具将 ONNX 模型转换为昇腾支持的.om格式
```bash
atc --model=best.onnx --output=yolo26n --soc_version=Ascend310
```
### 环境配置：使用官方Ubuntu镜像即可

### 推理部署：编写Python脚本，调用.om模型进行实时推理，支持摄像头视频流输入


## 许可证
本项目仅供学习交流使用，未经许可不得用于商业用途。

## 注意事项
香橙派 AIpro 需要 root 权限运行硬件相关代码
