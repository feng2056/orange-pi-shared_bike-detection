# 导出YOLO26n为ONNX格式（适配单类别 共享单车）
from ultralytics import YOLO
import os


def export_yolo26n_onnx():
    """导出YOLO26n模型为ONNX格式"""

    # 直接写你训练好的模型路径（100%匹配你现在的文件）
    model_path = r"D:\桌面\cycledata\scripts\runs\detect\train-2\weights\best.pt"

    if not os.path.exists(model_path):
        print(f"找不到YOLO26n模型文件: {model_path}")
        return

    # 加载模型
    print(f"加载YOLO26n模型: {model_path}")
    model = YOLO(model_path)

    # 导出ONNX（YOLO26n 昇腾/香橙派 最优参数）
    print("导出ONNX格式...")
    model.export(
        format='onnx',
        imgsz=640,
        opset=17,  # YOLO26n 必须用 17，兼容性最好
        simplify=True,  # 昇腾 NPU 必须开
        dynamic=False,
        half=False,
        batch=1,
    )

    print("YOLO26n ONNX导出完成！")
    print("输出文件：best.onnx")
    print("可直接转 OM 模型在香橙派AIpro运行")


if __name__ == '__main__':
    export_yolo26n_onnx()