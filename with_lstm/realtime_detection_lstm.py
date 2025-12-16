import cv2
import numpy as np
import mediapipe as mp
import os
import time
from threading import Thread
from collections import deque
import pygame
import torch
import torch.nn as nn

# MediaPipe 初始化
mp_facemesh = mp.solutions.face_mesh

# 关键点索引（与训练时一致）
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = {"left": 78, "right": 308, "upper": 13, "lower": 14}
LEFT_EYEBROW_MID = 66
LEFT_EYE_TOP = 159
RIGHT_EYEBROW_MID = 296
RIGHT_EYE_TOP = 386

# 时序配置
SEQUENCE_LENGTH = 30  # 使用30帧作为一个序列（约1秒，假设30fps）
MIN_SEQUENCE_LENGTH = 15  # 最少需要15帧才开始预测

# 警告配置
DROWSY_TIME_THRESHOLD = 2.0  # 轻度困倦持续2秒触发警告
SEVERE_DROWSY_TIME_THRESHOLD = 1.0  # 重度困倦持续1秒触发警告
DISTRACTION_TIME_THRESHOLD = 3.0  # 分心持续3秒触发警告
ALERT_REPEAT_INTERVAL = 3.0  # 警告重复间隔

# 类别标签
CLASS_NAMES = ["awake", "distracted", "light_drowsy", "severe_drowsy"]
CLASS_COLORS = {
    "awake": (0, 255, 0),  # 绿色
    "distracted": (0, 165, 255),  # 橙色
    "light_drowsy": (0, 255, 255),  # 黄色
    "severe_drowsy": (0, 0, 255),  # 红色
}


# PyTorch LSTM 模型定义
class LSTMDrowsinessModel(nn.Module):
    """LSTM 疲劳检测模型"""

    def __init__(
        self,
        input_size=7,
        hidden_size=64,
        num_layers=2,
        num_classes=4,
        dropout=0.3,
        bidirectional=False,
    ):
        super(LSTMDrowsinessModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)
        # 根据是否双向调整全连接层输入大小
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        out = self.dropout(last_output)
        out = self.fc(out)
        return out


def get_head_pose(landmarks, img_w, img_h):
    """计算头部姿态 (Pitch, Yaw, Roll)"""
    face_2d = []
    face_3d = []
    keypoints = [1, 152, 33, 263, 61, 291]

    for idx in keypoints:
        lm = landmarks[idx]
        x, y = int(lm.x * img_w), int(lm.y * img_h)
        face_2d.append([x, y])
        face_3d.append([x, y, lm.z])

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    focal_length = 1 * img_w
    cam_matrix = np.array(
        [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
    )
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    success, rot_vec, trans_vec = cv2.solvePnP(
        face_3d, face_2d, cam_matrix, dist_matrix
    )
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return [angles[0] * 360, angles[1] * 360, angles[2] * 360]


def eye_aspect_ratio(pts):
    """计算眼睛纵横比 (EAR)"""
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def eye_circularity(pts):
    """计算眼睛圆度"""
    perimeter = sum([np.linalg.norm(pts[i] - pts[(i + 1) % 6]) for i in range(6)])
    height = (np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])) / 2
    width = np.linalg.norm(pts[0] - pts[3])
    area = np.pi * (height / 2) * (width / 2)
    return (4 * np.pi * area) / (perimeter**2 + 1e-6)


def mouth_aspect_ratio(lm, h, w):
    """计算嘴部纵横比 (MAR)"""
    l = np.array([lm[MOUTH_IDX["left"]].x * w, lm[MOUTH_IDX["left"]].y * h])
    r = np.array([lm[MOUTH_IDX["right"]].x * w, lm[MOUTH_IDX["right"]].y * h])
    u = np.array([lm[MOUTH_IDX["upper"]].x * w, lm[MOUTH_IDX["upper"]].y * h])
    d = np.array([lm[MOUTH_IDX["lower"]].x * w, lm[MOUTH_IDX["lower"]].y * h])
    horizontal = np.linalg.norm(l - r)
    vertical = np.linalg.norm(u - d)
    return vertical / (horizontal + 1e-6)


def compute_features(landmarks, frame_width, frame_height):
    """
    计算所有7个特征: [EAR, MAR, Circularity, Brow Distance, Pitch, Yaw, Roll]

    Returns:
        features: [ear, mar, circ, brow_dist, pitch, yaw, roll]
        coords: 用于可视化的坐标字典
    """
    lm = landmarks
    w, h = frame_width, frame_height

    def get_pts(idx_list):
        return np.array(
            [[lm[i].x * w, lm[i].y * h] for i in idx_list], dtype=np.float32
        )

    def get_pt(idx):
        return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)

    # 1. EAR
    left_pts = get_pts(LEFT_EYE)
    right_pts = get_pts(RIGHT_EYE)
    left_ear = eye_aspect_ratio(left_pts)
    right_ear = eye_aspect_ratio(right_pts)
    ear = float((left_ear + right_ear) / 2.0)

    # 2. MAR
    mar = float(mouth_aspect_ratio(lm, h, w))

    # 3. Circularity
    left_circ = eye_circularity(left_pts)
    right_circ = eye_circularity(right_pts)
    circ = float((left_circ + right_circ) / 2.0)

    # 4. Eye-Eyebrow Distance (归一化)
    l_brow = get_pt(LEFT_EYEBROW_MID)
    l_eye = get_pt(LEFT_EYE_TOP)
    l_dist = np.linalg.norm(l_brow - l_eye)

    r_brow = get_pt(RIGHT_EYEBROW_MID)
    r_eye = get_pt(RIGHT_EYE_TOP)
    r_dist = np.linalg.norm(r_brow - r_eye)

    l_eye_width = np.linalg.norm(left_pts[0] - left_pts[3])
    r_eye_width = np.linalg.norm(right_pts[0] - right_pts[3])

    brow_dist = (l_dist / (l_eye_width + 1e-6) + r_dist / (r_eye_width + 1e-6)) / 2.0

    # 5. Head Pose
    pitch, yaw, roll = get_head_pose(lm, w, h)

    # 用于可视化的坐标
    coords = {
        "left_eye": left_pts,
        "right_eye": right_pts,
        "left_ear": left_ear,
        "right_ear": right_ear,
        "mouth": {
            "left": get_pt(MOUTH_IDX["left"]),
            "right": get_pt(MOUTH_IDX["right"]),
            "upper": get_pt(MOUTH_IDX["upper"]),
            "lower": get_pt(MOUTH_IDX["lower"]),
        },
        "mar": mar,
    }

    return [ear, mar, circ, brow_dist, pitch, yaw, roll], coords


def draw_landmarks(frame, coords):
    """绘制眼睛和嘴部关键点"""
    # 绘制眼睛关键点 (蓝色)
    for pts in [coords["left_eye"], coords["right_eye"]]:
        for pt in pts:
            cv2.circle(frame, tuple(pt.astype(int)), 2, (255, 0, 0), -1)

    # 绘制嘴部关键点 (绿色)
    mouth_pts = coords["mouth"]
    for key, pt in mouth_pts.items():
        cv2.circle(frame, tuple(pt.astype(int)), 3, (0, 255, 0), -1)

    # 绘制嘴部连线
    left = mouth_pts["left"].astype(int)
    right = mouth_pts["right"].astype(int)
    upper = mouth_pts["upper"].astype(int)
    lower = mouth_pts["lower"].astype(int)

    cv2.line(frame, tuple(left), tuple(right), (0, 255, 0), 1)
    cv2.line(frame, tuple(upper), tuple(lower), (0, 255, 0), 1)


def play_alert_sound(audio_path):
    """在独立线程中播放警告音"""
    try:
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"播放警告音失败: {e}")


class StateTracker:
    """状态跟踪器，管理不同状态的持续时间和警报"""

    def __init__(self):
        self.current_state = "awake"
        self.state_start_time = None
        self.last_alert_time = 0
        self.in_alert_mode = False

    def update(self, new_state, current_time):
        """更新状态"""
        if new_state != self.current_state:
            self.current_state = new_state
            self.state_start_time = current_time
            self.in_alert_mode = False

        return self.get_duration(current_time)

    def get_duration(self, current_time):
        """获取当前状态持续时间"""
        if self.state_start_time is None:
            self.state_start_time = current_time
            return 0
        return current_time - self.state_start_time

    def should_alert(self, current_time, threshold):
        """判断是否应该触发警报"""
        duration = self.get_duration(current_time)

        # 首次警报
        if duration >= threshold and not self.in_alert_mode:
            self.in_alert_mode = True
            self.last_alert_time = current_time
            return True

        # 重复警报
        if (
            self.in_alert_mode
            and (current_time - self.last_alert_time) >= ALERT_REPEAT_INTERVAL
        ):
            self.last_alert_time = current_time
            return True

        return False

    def reset(self):
        """重置状态"""
        self.state_start_time = None
        self.in_alert_mode = False


def main():
    # 初始化 pygame mixer
    pygame.mixer.init()

    # 警告音频路径
    alert_audio = r"video_dataset/12月15日.WAV"
    if not os.path.exists(alert_audio):
        print(f"警告: 未找到音频文件 {alert_audio}")
        alert_audio = None

    # 加载标准化参数 (新增)
    norm_params_path = "normalization_params.npy"
    norm_mean = None
    norm_std = None
    if os.path.exists(norm_params_path):
        try:
            norm_params = np.load(norm_params_path, allow_pickle=True).item()
            norm_mean = norm_params["mean"]
            norm_std = norm_params["std"]
            print("已加载标准化参数，将对输入进行归一化处理")
        except Exception as e:
            print(f"加载标准化参数失败: {e}")
    else:
        print(
            "警告: 未找到 normalization_params.npy，预测可能不准确！请确保已运行训练脚本。"
        )

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载 LSTM 模型
    model_path = "with_lstm/best_lstm_model.pth"
    if not os.path.exists(model_path):
        print(f"错误: 未找到模型文件 {model_path}")
        return

    try:
        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # 检查checkpoint格式
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # 从checkpoint中获取模型参数
            input_size = checkpoint.get("input_size", 7)
            hidden_size = checkpoint.get("hidden_size", 64)
            num_layers = checkpoint.get("num_layers", 2)
            num_classes = checkpoint.get("num_classes", 4)
            bidirectional = checkpoint.get("bidirectional", False)

            print(
                f"模型配置: input_size={input_size}, hidden_size={hidden_size}, "
                f"num_layers={num_layers}, num_classes={num_classes}, bidirectional={bidirectional}"
            )

            # 创建模型实例
            model = LSTMDrowsinessModel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=0.3,
                bidirectional=bidirectional,
            )

            # 加载模型权重
            model.load_state_dict(checkpoint["model_state_dict"])

            # 显示训练信息
            if "epoch" in checkpoint:
                print(f"模型训练轮次: {checkpoint['epoch']}")
            if "val_acc" in checkpoint:
                print(f"验证准确率: {checkpoint['val_acc']:.2%}")
        else:
            # 直接加载state_dict（假设是单向LSTM）
            model = LSTMDrowsinessModel(
                input_size=7,
                hidden_size=64,
                num_layers=2,
                num_classes=4,
                dropout=0.3,
                bidirectional=False,
            )
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()  # 设置为评估模式

        print(f"已加载 PyTorch LSTM 模型: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        import traceback

        traceback.print_exc()
        return

    # 打开视频
    cap = cv2.VideoCapture(r"video_dataset\daraset.mp4")

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("错误: 无法打开视频文件")
        return

    # 特征序列缓冲区（滑动窗口）
    feature_buffer = deque(maxlen=SEQUENCE_LENGTH)

    # 状态跟踪器
    state_tracker = StateTracker()

    with mp_facemesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            (h, w) = frame.shape[:2]
            target_height = 480
            ratio = target_height / float(h)
            new_width = int(w * ratio)

            frame = cv2.resize(frame, (new_width, target_height))

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgH, imgW, _ = frame.shape
            current_time = time.time()

            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # 计算7个特征
                features, coords = compute_features(landmarks, imgW, imgH)

                # 添加到缓冲区
                feature_buffer.append(features)

                # 绘制关键点
                draw_landmarks(frame, coords)

                # 当有足够的帧时进行预测
                prediction_text = ""
                status_color = (0, 255, 0)
                predicted_class = "awake"

                if len(feature_buffer) >= MIN_SEQUENCE_LENGTH:
                    # 准备输入序列
                    sequence = np.array(list(feature_buffer))

                    # 如果不足 SEQUENCE_LENGTH，用最后一帧填充
                    if len(sequence) < SEQUENCE_LENGTH:
                        padding = np.repeat(
                            [sequence[-1]], SEQUENCE_LENGTH - len(sequence), axis=0
                        )
                        sequence = np.vstack([sequence, padding])

                    # 数据标准化 (新增关键步骤)
                    if norm_mean is not None and norm_std is not None:
                        sequence = (sequence - norm_mean) / norm_std

                    # 转换为 PyTorch tensor
                    sequence_tensor = (
                        torch.FloatTensor(sequence).unsqueeze(0).to(device)
                    )
                    # sequence_tensor shape: (1, SEQUENCE_LENGTH, 7)

                    # 预测
                    with torch.no_grad():
                        outputs = model(sequence_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        predictions = probabilities.cpu().numpy()[0]

                    # 获取预测类别和概率
                    pred_idx = np.argmax(predictions)
                    predicted_class = CLASS_NAMES[pred_idx]
                    confidence = predictions[pred_idx]

                    # --- 规则修正 (Heuristic Correction) ---
                    # 解决分心和困倦混淆的问题
                    current_ear = features[0]  # EAR是第0个特征

                    # 规则1: 如果眼睛闭合 (EAR很低)，强制判定为困倦，即使头部有动作(分心)
                    # 这里的 0.18 是一个经验阈值，可以根据实际情况微调
                    if current_ear < 0.18 and predicted_class == "distracted":
                        predicted_class = "severe_drowsy"
                        confidence = 1.0  # 规则强制
                        prediction_text = f"{predicted_class.upper()} (Eye Closed)"

                    # 规则2: 如果眼睛睁得很大，不应该是重度困倦
                    elif current_ear > 0.25 and predicted_class == "severe_drowsy":
                        predicted_class = "light_drowsy"  # 降级为轻度或清醒
                        if confidence < 0.8:  # 如果置信度不高，甚至可能是清醒
                            predicted_class = "awake"
                        prediction_text = f"{predicted_class.upper()} (Eye Open)"
                    else:
                        prediction_text = (
                            f"{predicted_class.upper()} ({confidence:.2%})"
                        )
                    # -------------------------------------

                    status_color = CLASS_COLORS[predicted_class]

                    # 更新状态跟踪
                    duration = state_tracker.update(predicted_class, current_time)

                    # 判断是否需要警报
                    should_play_alert = False

                    if predicted_class == "severe_drowsy":
                        if state_tracker.should_alert(
                            current_time, SEVERE_DROWSY_TIME_THRESHOLD
                        ):
                            should_play_alert = True
                    elif predicted_class == "light_drowsy":
                        if state_tracker.should_alert(
                            current_time, DROWSY_TIME_THRESHOLD
                        ):
                            should_play_alert = True
                    elif predicted_class == "distracted":
                        if state_tracker.should_alert(
                            current_time, DISTRACTION_TIME_THRESHOLD
                        ):
                            should_play_alert = True
                    else:
                        state_tracker.reset()

                    # 播放警报
                    if should_play_alert and alert_audio is not None:
                        if not pygame.mixer.music.get_busy():
                            Thread(
                                target=play_alert_sound,
                                args=(alert_audio,),
                                daemon=True,
                            ).start()

                    # 显示持续时间
                    if duration > 0.5 and predicted_class != "awake":
                        duration_text = (
                            f"{predicted_class.upper()} Duration: {duration:.1f}s"
                        )
                        cv2.putText(
                            frame,
                            duration_text,
                            (10, imgH - 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            status_color,
                            2,
                        )

                        # 严重困倦时闪烁警告
                        if (
                            predicted_class == "severe_drowsy"
                            and int(current_time * 4) % 2 == 0
                        ):
                            overlay = frame.copy()
                            cv2.rectangle(
                                overlay, (0, 0), (imgW, imgH), (0, 0, 255), 40
                            )
                            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

                # 显示特征值
                y_offset = 30
                feature_names = ["EAR", "MAR", "Circ", "Brow", "Pitch", "Yaw", "Roll"]
                for i, (name, value) in enumerate(zip(feature_names, features)):
                    cv2.putText(
                        frame,
                        f"{name}: {value:.3f}",
                        (10, y_offset + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

                # 显示预测结果
                if prediction_text:
                    cv2.putText(
                        frame,
                        prediction_text,
                        (10, imgH - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        status_color,
                        3,
                    )

                # 显示概率分布
                if len(feature_buffer) >= MIN_SEQUENCE_LENGTH:
                    prob_y = imgH - 150
                    for i, (cls, prob) in enumerate(zip(CLASS_NAMES, predictions)):
                        bar_width = int(prob * 200)
                        cv2.rectangle(
                            frame,
                            (imgW - 220, prob_y + i * 30),
                            (imgW - 220 + bar_width, prob_y + i * 30 + 20),
                            CLASS_COLORS[cls],
                            -1,
                        )
                        cv2.putText(
                            frame,
                            f"{cls}: {prob:.2%}",
                            (imgW - 210, prob_y + i * 30 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 255),
                            1,
                        )

            else:
                cv2.putText(
                    frame,
                    "No face detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                feature_buffer.clear()
                state_tracker.reset()

            cv2.imshow("LSTM Drowsiness Detection", frame)

            if cv2.waitKey(1000 // SEQUENCE_LENGTH) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()


if __name__ == "__main__":
    main()
