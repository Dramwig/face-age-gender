import enum
from ultralytics import YOLO
from age_gender_ssrnet.SSRNET_model import SSR_net_general, SSR_net


class YOLOModel(enum.Enum):
    YOLOv8n_Face = "model/yolov8n-face-lindevs.pt"
    YOLOv8s_Face = "model/yolov8s-face-lindevs.pt"
    
def load_yolo_model(model_name: YOLOModel):
    # 获取模型文件路径
    model_path = model_name.value 
    # 加载模型
    model = YOLO(model_path)
    return model

def load_ssr_model():
    face_size = 64
    face_padding_ratio = 0.10
    # Default parameters for SSR-Net
    stage_num = [3, 3, 3]
    lambda_local = 1
    lambda_d = 1
    # Initialize gender net
    gender_net = SSR_net_general(face_size, stage_num, lambda_local, lambda_d)()
    gender_net.load_weights('age_gender_ssrnet/ssrnet_gender_3_3_3_64_1.0_1.0.h5')
    # Initialize age net
    age_net = SSR_net(face_size, stage_num, lambda_local, lambda_d)()
    age_net.load_weights('age_gender_ssrnet/ssrnet_age_3_3_3_64_1.0_1.0.h5')
    return gender_net, age_net