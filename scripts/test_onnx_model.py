# test_onnx_model.py
import cv2
import numpy as np
import onnxruntime as ort
import time
import os
from pathlib import Path


def test_onnx_model():
    """测试ONNX模型"""
    print("🧪 测试ONNX模型 (best.onnx)")
    print("=" * 50)

    # 1. 检查模型文件
    onnx_path = "best.onnx"
    if not os.path.exists(onnx_path):
        # 尝试在其他位置查找
        possible_paths = [
            "best.onnx",
            "runs/detect/train2/weights/best.onnx",
            "runs/train/shared_bicycle_detection/weights/best.onnx"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                onnx_path = path
                break

    if not os.path.exists(onnx_path):
        print(f"❌ 找不到ONNX模型文件")
        print("请确保best.onnx文件存在")
        return None

    print(f"📦 加载ONNX模型: {onnx_path}")

    # 2. 加载ONNX模型
    try:
        ort_session = ort.InferenceSession(onnx_path)
        print("✅ ONNX模型加载成功")

        # 打印模型信息
        print("\n📋 模型信息:")
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        print(f"  输入名称: {input_name}")
        print(f"  输出名称: {output_name}")
        print(f"  输入形状: {ort_session.get_inputs()[0].shape}")
        print(f"  输出形状: {ort_session.get_outputs()[0].shape}")
    except Exception as e:
        print(f"❌ ONNX模型加载失败: {e}")
        return None

    # 3. 预处理函数
    def preprocess(image_path):
        """预处理图像"""
        img = cv2.imread(image_path)
        if img is None:
            return None, None, None, None

        orig_h, orig_w = img.shape[:2]

        # 保持宽高比resize到640x640
        target_size = 640
        scale = min(target_size / orig_h, target_size / orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)

        # Resize
        resized = cv2.resize(img, (new_w, new_h))

        # 转换为RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 填充到640x640（左上角填充）
        padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = rgb

        # 转换为CHW格式并归一化
        input_data = padded.transpose(2, 0, 1)  # HWC to CHW
        input_data = np.ascontiguousarray(input_data, dtype=np.float32)
        input_data = input_data / 255.0  # 归一化

        # 添加batch维度
        input_data = np.expand_dims(input_data, axis=0)

        return input_data, scale, img, (orig_h, orig_w), (new_h, new_w)

    # 4. 后处理函数
    def postprocess(predictions, scale, orig_img, orig_shape, resized_shape, conf_threshold=0.25):
        """后处理"""
        if predictions is None:
            return [], []

        preds = predictions[0]  # [5, 8400] 或 [84, 8400]

        # 检查输出形状
        if preds.shape[0] == 84:
            # 多类别输出，需要提取前5个值
            # 对于单类别，我们只需要前5个值: [4个坐标 + 1个置信度]
            preds = preds[:5, :]
        elif preds.shape[0] == 5:
            # 单类别输出
            pass
        else:
            print(f"⚠️  未知的输出形状: {preds.shape}")
            return [], []

        boxes = []
        confidences = []

        for i in range(preds.shape[1]):
            conf = preds[4, i]

            if conf < conf_threshold:
                continue

            # 获取坐标
            cx = preds[0, i]  # 中心点x (0-1)
            cy = preds[1, i]  # 中心点y (0-1)
            bw = preds[2, i]  # 宽度 (0-1)
            bh = preds[3, i]  # 高度 (0-1)

            # 转换为640x640像素坐标
            x1_640 = (cx - bw / 2) * 640
            y1_640 = (cy - bh / 2) * 640
            x2_640 = (cx + bw / 2) * 640
            y2_640 = (cy + bh / 2) * 640

            # 转换到原始图像坐标（左上角填充，偏移为0）
            x1 = int(x1_640 / scale)
            y1 = int(y1_640 / scale)
            x2 = int(x2_640 / scale)
            y2 = int(y2_640 / scale)

            # 确保在图像范围内
            orig_h, orig_w = orig_shape
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            # 过滤小框
            if (x2 - x1) >= 20 and (y2 - y1) >= 20:
                boxes.append([x1, y1, x2, y2])
                confidences.append(float(conf))

        return boxes, confidences

    # 5. 测试图像
    test_images = []

    # 优先使用test.jpg
    if os.path.exists("test.jpg"):
        test_images.append("test.jpg")

    # 如果没有test.jpg，使用验证集图像
    if not test_images:
        val_dir = Path("dataset/images/val")
        if val_dir.exists():
            val_images = list(val_dir.glob("*.jpg"))[:3]
            test_images = [str(img) for img in val_images]

    if not test_images:
        print("❌ 找不到测试图像")
        return None

    results = {}

    for img_path in test_images:
        print(f"\n🔍 测试图像: {img_path}")

        # 预处理
        start_time = time.time()
        input_data, scale, orig_img, orig_shape, resized_shape = preprocess(img_path)
        if input_data is None:
            continue

        # 推理
        predictions = ort_session.run(None, {input_name: input_data})[0]
        inference_time = time.time() - start_time

        # 后处理
        boxes, confidences = postprocess(predictions, scale, orig_img, orig_shape, resized_shape)

        # NMS
        final_boxes = []
        final_confidences = []

        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
            if indices is not None:
                for i in indices.flatten():
                    final_boxes.append(boxes[i])
                    final_confidences.append(confidences[i])

        # 保存结果图像
        result_img = orig_img.copy()
        for i, (box, conf) in enumerate(zip(final_boxes, final_confidences)):
            x1, y1, x2, y2 = box
            color = (0, 0, 255)  # 红色，便于区分

            # 绘制框
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(result_img, f'{conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 保存结果
        output_path = f"onnx_result_{Path(img_path).stem}.jpg"
        cv2.imwrite(output_path, result_img)

        print(f"✅ 检测到 {len(final_boxes)} 个共享单车")
        print(f"📊 推理时间: {inference_time * 1000:.1f}ms")
        print(f"💾 结果保存到: {output_path}")

        results[img_path] = {
            'boxes': final_boxes,
            'confidences': final_confidences,
            'count': len(final_boxes),
            'time': inference_time,
            'output_path': output_path
        }

    return results


def print_onnx_summary(results):
    """打印ONNX模型测试总结"""
    if not results:
        return

    print("\n" + "=" * 50)
    print("📊 ONNX模型测试总结")
    print("=" * 50)

    total_boxes = 0
    total_images = len(results)
    total_time = 0

    for img_path, result in results.items():
        boxes_count = result['count']
        inference_time = result['time']

        print(f"📷 {Path(img_path).name}:")
        print(f"  检测数量: {boxes_count}")
        print(f"  推理时间: {inference_time * 1000:.1f}ms")

        if boxes_count > 0:
            print(f"  置信度范围: {min(result['confidences']):.3f} - {max(result['confidences']):.3f}")

        total_boxes += boxes_count
        total_time += inference_time

    if total_images > 0:
        avg_boxes = total_boxes / total_images
        avg_time = total_time / total_images

        print(f"\n📈 总体统计:")
        print(f"  测试图像数: {total_images}")
        print(f"  总检测框数: {total_boxes}")
        print(f"  平均每张图像: {avg_boxes:.1f} 个")
        print(f"  平均推理时间: {avg_time * 1000:.1f}ms")


if __name__ == "__main__":
    onnx_results = test_onnx_model()
    print_onnx_summary(onnx_results)