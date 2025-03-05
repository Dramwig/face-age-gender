import cv2
import numpy as np
from ultralytics import YOLO
import time
from utils import load_yolo_model, YOLOModel, load_ssr_model

def predict_gender(faces, gender_net, age_net):
    face_size = 64
    blob = np.empty((len(faces), face_size, face_size, 3))
    for i, face_bgr in enumerate(faces):
        blob[i, :, :, :] = cv2.resize(face_bgr, (64, 64))
        blob[i, :, :, :] = cv2.normalize(blob[i, :, :, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # Predict gender and age
    genders = gender_net.predict(blob)
    ages = age_net.predict(blob)
    #  Construct labels
    labels = ['{},{}'.format('Male' if (gender >= 0.5) else 'Female', int(age)) for (gender, age) in zip(genders, ages)]
    return labels

def process_video(video_path, output_path):
    # 加载YOLO模型
    model = load_yolo_model(YOLOModel.YOLOv8n_Face)
    gender_net, age_net = load_ssr_model()
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 如果视频文件不存在，则报错
    if not cap.isOpened():
        raise ValueError(f"视频文件 {video_path} 不存在")
    
    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # 使用YOLO进行人脸检测
        results = model(frame)
        
        # 处理检测结果
        rectangles = []
        faces = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 提取人脸区域
                face = frame[y1:y2, x1:x2]
                
                rectangles.append((x1, y1, x2, y2))
                faces.append(face)
                
        labels = predict_gender(faces, gender_net, age_net)

        for rectangle, label in zip(rectangles, labels):
            x1, y1, x2, y2 = rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # 写入处理后的帧
        out.write(frame)
        
        # 打印处理进度
        print(f"处理帧: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
        
        # if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) > 200:
        #     break   
    
    # 释放资源
    cap.release()
    out.release()
    print(f"处理完成，结果保存到: {output_path}")

if __name__ == "__main__":
    video_path = "video/白宫吵架大会2.mp4"  # 使用正确的视频文件名
    output_path = 'outputs/output.mp4'
    process_video(video_path, output_path)
