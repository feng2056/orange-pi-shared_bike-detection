# -*- coding: utf-8 -*-
import cv2
import numpy as np
from ais_bench.infer.interface import InferSession


# 类别定义
CLASSES = {
    0: 'shared_bicycle'
}

# 置信度阈值
CONFIDENCE = 0.25
# NMS 的 IoU 阈值
IOU = 0.3

# 为每个类别分配随机颜色
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """绘制检测框和标签，增加边界值保护"""
    try:
        label = "{} {:.2f}".format(CLASSES[class_id], confidence)
        color = colors[class_id]
        
        # 确保坐标在图像范围内
        h, w = img.shape[:2]
        x = max(0, min(int(x), w))
        y = max(0, min(int(y), h))
        x_plus_w = max(0, min(int(x_plus_w), w))
        y_plus_h = max(0, min(int(y_plus_h), h))
        
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_width, label_height = label_size
        label_x = x
        label_y = y - 10 if y - 10 > label_height else y + 10
        
        # 确保标签框不越界
        label_y = max(label_height, min(label_y, h - label_height))
        label_x = max(0, min(label_x, w - label_width))
        
        cv2.rectangle(img, (label_x, label_y - label_height),
                      (label_x + label_width, label_y + label_height), color, cv2.FILLED)
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    except Exception as e:
        print(f"绘制边界框出错: {e}")


def main(session, original_image):
    """推理主函数，修复数据类型和索引问题"""
    try:
        height, width, _ = original_image.shape
        length = max(height, width)
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image
        scale = length / 640.0  # 确保浮点型
        
        # 预处理图像
        blob = cv2.dnn.blobFromImage(
            image, 
            scalefactor=1.0 / 255, 
            size=(640, 640), 
            swapRB=True,
            crop=False  # 显式指定不裁剪，避免默认行为差异
        )
        
        # 推理
        output_data = session.infer(feeds=blob, mode="static")
        outputs = output_data[0] if isinstacnce(output_data, (list, tuple)) else output_data
        outputs = np.array([outputs])

        # 兼容输出的不同维度格式
        if isinstance(outputs, list) and len(outputs) > 0:
            outputs = np.array(outputs[0])
        if len(outputs.shape) > 2:
            outputs = np.array([cv2.transpose(outputs[0])])
        
        rows = outputs.shape[1] if len(outputs.shape) >= 2 else 0

        boxes = []
        scores = []
        class_ids = []

        for i in range(rows):
            # 确保索引不越界
            if i >= outputs.shape[1]:
                continue
            classes_scores = outputs[0][i][4:] if len(outputs[0][i]) > 4 else []
            if len(classes_scores) == 0:
                continue
                
            # 获取最大置信度和类别
            minScore, maxScore, minClassLoc, (x, maxClassIndex) = cv2.minMaxLoc(classes_scores)
            if maxScore >= CONFIDENCE:
                # 计算检测框坐标，增加非负保护
                cx = outputs[0][i][0]
                cy = outputs[0][i][1]
                w = outputs[0][i][2]
                h = outputs[0][i][3]
                
                x1 = max(0, (cx - w / 2) * scale)
                y1 = max(0, (cy - h / 2) * scale)
                box_w = max(1, w * scale)  # 避免宽度/高度为0
                box_h = max(1, h * scale)
                
                box = [x1, y1, box_w, box_h]
                boxes.append(box)
                scores.append(float(maxScore))  # 确保浮点型
                class_ids.append(int(maxClassIndex))  # 确保整型

        # NMS非极大值抑制，兼容不同OpenCV版本的返回格式
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE, IOU)
        detections = []

        # 处理NMS返回的索引格式（可能是二维数组）
        if len(result_boxes) > 0:
            if result_boxes.ndim == 2:
                result_boxes = result_boxes.flatten()  # 展平为一维数组
            
            for i in range(len(result_boxes)):
                try:
                    index = result_boxes[i]
                    # 索引边界保护
                    if index < 0 or index >= len(boxes):
                        continue
                        
                    box = boxes[index]
                    detection = {
                        "class_id": class_ids[index],
                        "class_name": CLASSES.get(class_ids[index], "unknown"),
                        "confidence": scores[index],
                        "box": box,
                        "scale": scale,
                    }
                    detections.append(detection)
                    
                    # 绘制检测框
                    draw_bounding_box(
                        original_image,
                        class_ids[index],
                        scores[index],
                        round(box[0]),
                        round(box[1]),
                        round(box[0] + box[2]),
                        round(box[1] + box[3])
                    )
                except Exception as e:
                    print(f"处理检测框{i}出错: {e}")
        
        return original_image, detections
    except Exception as e:
        print(f"推理主函数出错: {e}")
        return original_image, []


def compute_iou(box1, box2):
    """计算IoU，增加非负保护"""
    try:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # 确保坐标为非负
        x1, y1, w1, h1 = max(0, x1), max(0, y1), max(0, w1), max(0, h1)
        x2, y2, w2, h2 = max(0, x2), max(0, y2), max(0, w2), max(0, h2)
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    except Exception as e:
        print(f"计算IoU出错: {e}")
        return 0.0


def evaluate_density(image, detections,
                     area_weight=0.7,
                     count_weight=0.3,
                     iou_weight=0.5,
                     conf_weight=0.1,
                     density_threshold=0.5):
    """密度评估，增加异常保护"""
    try:
        img_h, img_w = image.shape[:2]
        img_area = max(1, img_h * img_w)  # 避免除以0
        num_boxes = len(detections)

        if num_boxes == 0:
            return "低密度", 0.0, {"num":0, "cover":0, "avg_iou":0, "avg_conf":0, "score":0.0}

        count_density = num_boxes / img_area
        total_cover = 0.0
        conf_sum = 0.0
        boxes = []
        
        for det in detections:
            try:
                x, y, w, h = det["box"]
                total_cover += w * h
                conf_sum += det["confidence"]
                boxes.append([x, y, w, h])
            except Exception as e:
                print(f"处理检测结果出错: {e}")
                continue
        
        cover_ratio = total_cover / img_area
        avg_conf = conf_sum / num_boxes if num_boxes > 0 else 0.0

        avg_iou = 0.0
        if num_boxes > 1:
            iou_sum = 0.0
            pair_count = 0
            for i in range(num_boxes):
                for j in range(i+1, num_boxes):
                    iou = compute_iou(boxes[i], boxes[j])
                    iou_sum += iou
                    pair_count += 1
            avg_iou = iou_sum / pair_count if pair_count > 0 else 0.0

        # 计算密度分数，权重归一化保护
        total_weight = area_weight + count_weight + iou_weight + conf_weight
        if total_weight <= 0:
            total_weight = 1.0
        
        density_score = (
            area_weight * cover_ratio +
            count_weight * (count_density * 10000) +
            iou_weight * avg_iou +
            conf_weight * (1 - avg_conf)
        ) / total_weight  # 归一化权重
        
        level = "高密度" if density_score > density_threshold else "低密度"
        details = {
            "num": num_boxes,
            "cover": round(cover_ratio, 6),
            "avg_iou": round(avg_iou, 6),
            "avg_conf": round(avg_conf, 6),
            "score": round(density_score, 6)
        }
        return level, density_score, details
    except Exception as e:
        print(f"密度评估出错: {e}")
        return "低密度", 0.0, {"num":0, "cover":0, "avg_iou":0, "avg_conf":0, "score":0.0}


if __name__ == "__main__":
    # 主程序增加异常处理
    try:
        model_path = "shared_bicycle_yolo26n.om"
        # 创建推理会话，增加设备ID兼容性
        session = InferSession(device_id=0, model_path=model_path)

        # 打开摄像头，增加重试机制
        cap = None
        for cam_id in [0,1,2]:  # 尝试多个摄像头ID
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                print(f"成功打开摄像头 {cam_id}")
                break        
        if not cap or not cap.isOpened():
            print("无法打开任何摄像头")
            exit(1)

        # 设置摄像头参数，提高兼容性
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("开始定期检测（每2秒一次），按 'q' 键退出...")
        detect_interval = 2.0  # 检测间隔（秒）
        last_detect_time = 0


        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法获取画面，重试...")
                continue
                        
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            if current_time - last_detect_time >= detect_interval:
                # 推理（在图像副本上进行，避免影响原始帧的显示）
                draw_image, detections = main(session, frame.copy())

                # 密度评估
                level, score, details = evaluate_density(frame, detections)

                print("\n========== 密度评估结果 ==========")
                print(f"检测到共享单车数量: {details['num']}")
                print(f"面积覆盖率: {details['cover']:.3f}")
                print(f"平均框间IoU: {details['avg_iou']:.3f}")
                print(f"平均置信度: {details['avg_conf']:.3f}")
                print(f"综合密度分数: {details['score']:.3f}")
                print(f"最终结论: {level}")
                print("==================================\n")

                # 显示绘制了边界框的图像
                cv2.imshow("Detection", draw_image)
                last_detect_time = current_time

            # 按 'q' 退出，增加按键检测的稳定性
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # 按 'ESC' 也可退出
            elif key == 27:
                break

    except Exception as e:
        print(f"主程序出错: {e}")
    finally:
        # 确保资源释放
        if 'cap' in locals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("程序已退出，资源已释放")

