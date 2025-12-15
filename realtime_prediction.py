import cv2
import numpy as np
import mediapipe as mp
import pickle
import os

mp_facemesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 使用与 notebook 相同的关键点索引
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = {"left": 78, "right": 308, "upper": 13, "lower": 14}
LEFT_EYEBROW_MID = 66
LEFT_EYE_TOP = 159
RIGHT_EYEBROW_MID = 296
RIGHT_EYE_TOP = 386


def eye_aspect_ratio(pts):
    """计算眼睛纵横比 (EAR)"""
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def eye_circularity(pts):
    """计算眼睛圆度"""
    perimeter = (
        np.linalg.norm(pts[0] - pts[1])
        + np.linalg.norm(pts[1] - pts[2])
        + np.linalg.norm(pts[2] - pts[3])
        + np.linalg.norm(pts[3] - pts[4])
        + np.linalg.norm(pts[4] - pts[5])
        + np.linalg.norm(pts[5] - pts[0])
    )
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
    计算所有特征: EAR, MAR, Circularity, Brow Distance

    Returns:
        features: [ear, mar, circ, brow_dist]
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

    # 获取嘴部关键点坐标
    mouth_pts = {
        "left": get_pt(MOUTH_IDX["left"]),
        "right": get_pt(MOUTH_IDX["right"]),
        "upper": get_pt(MOUTH_IDX["upper"]),
        "lower": get_pt(MOUTH_IDX["lower"]),
    }

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

    # 用于可视化的坐标
    coords = {
        "left_eye": left_pts,
        "right_eye": right_pts,
        "left_ear": left_ear,
        "right_ear": right_ear,
        "mouth": mouth_pts,
        "mar": mar,
    }

    return [ear, mar, circ, brow_dist], coords


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

    cv2.line(frame, tuple(left), tuple(right), (0, 255, 0), 1)  # 水平线
    cv2.line(frame, tuple(upper), tuple(lower), (0, 255, 0), 1)  # 垂直线


def main():
    # 尝试加载训练好的模型
    model = None
    model_path = "drowsiness_model.pkl"
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            print(f"已加载模型: {model_path}")
        except:
            print(f"无法加载模型: {model_path}，将只显示特征值")
    else:
        print(f"未找到模型文件: {model_path}，将只显示特征值")

    cap = cv2.VideoCapture(r"video_dataset\2.mp4")

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

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgH, imgW, _ = frame.shape

            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # 计算特征
                features, coords = compute_features(landmarks, imgW, imgH)
                ear, mar, circ, brow_dist = features

                # 绘制眼睛关键点
                draw_landmarks(frame, coords)

                # 模型预测
                prediction_text = ""
                status_color = (0, 255, 0)  # 绿色

                if model is not None:
                    pred = model.predict([features])[0]
                    prob = model.predict_proba([features])[0]

                    if pred == 1:  # 困倦
                        prediction_text = f"DROWSY (Prob: {prob[1]:.2%})"
                        status_color = (0, 0, 255)  # 红色
                    else:  # 非困倦
                        prediction_text = f"ALERT (Prob: {prob[0]:.2%})"
                        status_color = (0, 255, 0)  # 绿色

                # 显示特征值
                y_offset = 30
                cv2.putText(
                    frame,
                    f"EAR: {ear:.3f}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                y_offset += 30
                cv2.putText(
                    frame,
                    f"MAR: {mar:.3f}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                y_offset += 30
                cv2.putText(
                    frame,
                    f"Circ: {circ:.3f}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                y_offset += 30
                cv2.putText(
                    frame,
                    f"Brow: {brow_dist:.3f}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
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

                # 在眼睛旁显示左右眼 EAR
                left_pt = coords["left_eye"][0].astype(int)
                right_pt = coords["right_eye"][0].astype(int)
                cv2.putText(
                    frame,
                    f"L: {coords['left_ear']:.3f}",
                    (left_pt[0] - 30, left_pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"R: {coords['right_ear']:.3f}",
                    (right_pt[0] - 30, right_pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

                # 在嘴部旁显示 MAR
                mouth_center = coords["mouth"]["right"].astype(int)
                cv2.putText(
                    frame,
                    f"MAR: {coords['mar']:.3f}",
                    (mouth_center[0] + 10, mouth_center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
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

            cv2.imshow("Drowsiness Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
