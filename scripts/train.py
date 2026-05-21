from ultralytics import YOLO
import yaml
import os
from pathlib import Path

def train_yolo26n():
    """训练YOLO26n模型（单类别：共享单车）"""
    # 1. 配置检查
    yaml_path = "../configs/shared_bicycle.yaml"
    if not os.path.exists(yaml_path):
        print(f"找不到配置文件: {yaml_path}")
        return None

    # 2. 加载YOLO26n预训练模型（核心修改点）
    print("加载YOLO26n预训练模型")
    model = YOLO("yolo26n.pt")  # 替换为yolo26n.pt

    # 3. 训练配置（基于原有参数优化，适配YOLO26n）
    print("开始训练YOLO26n模型（单类别共享单车）")
    results = model.train(
        data=yaml_path,          # 数据集配置文件
        epochs=100,              # 训练轮数（YOLO26n收敛更快，可酌情减至80）
        imgsz=640,               # 输入尺寸（必须与测试/导出一致）
        batch=16,                # 批次大小（根据GPU显存调整，CPU建议8）
        device='cpu',            # GPU使用0/1，CPU用'cpu'
        workers=4,               # 数据加载线程
        optimizer='AdamW',       # 优化器（YOLO26n推荐AdamW）
        lr0=0.001,               # 初始学习率（YOLO26n可稍降为0.0008）
        lrf=0.01,                # 最终学习率因子
        momentum=0.937,          # 动量
        weight_decay=0.0005,     # 权重衰减
        warmup_epochs=3,         # 热身轮数
        box=7.5,                 # 框损失权重
        cls=0.5,                 # 分类损失权重（单类别可稍提至0.8）
        dfl=1.5,                 # DFL损失权重
        save=True,               # 保存最佳模型
        save_period=10,          # 每10轮保存一次
        pretrained=True,         # 使用预训练权重
        single_cls=True,         # 强制单类别训练（关键！确保类别索引统一）
        cos_lr=True,             # 余弦学习率（YOLO26n推荐）
        val=True,                # 训练中验证
        plots=True,              # 生成训练可视化图表
        seed=42,                 # 固定随机种子
        deterministic=True,      # 确定性训练
        rect=False,              # 关闭矩形训练（单类别无需）
        label_smoothing=0.0,     # 标签平滑（单类别禁用）
        nbs=64,                  # 标称批次大小
    )

    print("YOLO26n训练完成！")
    print(f"模型保存路径: {model.ckpt_path}")
    return results

if __name__ == '__main__':
    # 确保工作目录正确
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    train_yolo26n()