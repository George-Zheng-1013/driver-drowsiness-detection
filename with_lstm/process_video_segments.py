import os
import cv2
import numpy as np
import mediapipe as mp
import re
from tqdm import tqdm
from collections import defaultdict

# ================= 配置区域 =================
DATASET_PATH = r"D:\HP\OneDrive\Desktop\学校\课程\专业课\神经网络\课程项目\driver-drowsiness-detection\archive\Driver Drowsiness Dataset (DDD)"
OUTPUT_NPY_DIR = "processed_features"

# 分类阈值
EAR_LIGHT_DROWSY_THRESHOLD = 0.20
EAR_SEVERE_DROWSY_THRESHOLD = 0.12

# 分心合成参数
DISTRACTION_RATIO = 0.3
YAW_DISTRACTION_RANGE = (25, 45)
PITCH_DISTRACTION_RANGE = (20, 35)

# 数据增强配置
ENABLE_AUGMENTATION = True  # 是否启用数据增强
AUGMENTATION_RATIO = 3
# ===========================================

# MediaPipe 初始化
mp_face_mesh = mp.solutions.face_mesh

# 关键点索引
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = {"left": 78, "right": 308, "upper": 13, "lower": 14}
LEFT_EYEBROW_MID = 66
LEFT_EYE_TOP = 159
RIGHT_EYEBROW_MID = 296
RIGHT_EYE_TOP = 386


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


def calculate_all_features(image, face_mesh):
    """提取单帧的7个特征 [EAR, MAR, Circ, Brow, Pitch, Yaw, Roll]"""
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None, None

    lm = results.multi_face_landmarks[0].landmark
    np_lm = np.array([[p.x * w, p.y * h] for p in lm])

    # 1. EAR
    left_pts = np_lm[LEFT_EYE]
    right_pts = np_lm[RIGHT_EYE]
    l_ear = eye_aspect_ratio(left_pts)
    r_ear = eye_aspect_ratio(right_pts)
    avg_ear = (l_ear + r_ear) / 2.0

    # 2. MAR
    mar = mouth_aspect_ratio(lm, h, w)

    # 3. Circularity
    l_circ = eye_circularity(left_pts)
    r_circ = eye_circularity(right_pts)
    circ = (l_circ + r_circ) / 2.0

    # 4. Brow Distance
    l_brow_dist = np.linalg.norm(np_lm[LEFT_EYEBROW_MID] - np_lm[LEFT_EYE_TOP])
    r_brow_dist = np.linalg.norm(np_lm[RIGHT_EYEBROW_MID] - np_lm[RIGHT_EYE_TOP])
    l_eye_width = np.linalg.norm(left_pts[0] - left_pts[3])
    r_eye_width = np.linalg.norm(right_pts[0] - right_pts[3])
    avg_brow = (
        l_brow_dist / (l_eye_width + 1e-6) + r_brow_dist / (r_eye_width + 1e-6)
    ) / 2.0

    # 5. Head Pose
    pitch, yaw, roll = get_head_pose(lm, w, h)

    features = [avg_ear, mar, circ, avg_brow, pitch, yaw, roll]

    return features, results.multi_face_landmarks[0]


def imread_unicode(img_path):
    """支持中文路径的图片读取"""
    from PIL import Image

    img_pil = Image.open(img_path)
    img_array = np.array(img_pil)
    # PIL 是 RGB, OpenCV 是 BGR
    if len(img_array.shape) == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array
    return img_bgr


def synthesize_distraction(features):
    """
    合成分心数据:修改头部姿态特征
    features: [EAR, MAR, Circ, Brow, Pitch, Yaw, Roll]
    """
    features_copy = features.copy()

    # 随机选择转头或低头
    if np.random.random() < 0.5:
        # 转头 (修改 Yaw, index=5)
        direction = np.random.choice([-1, 1])
        yaw_offset = np.random.uniform(*YAW_DISTRACTION_RANGE)
        features_copy[:, 5] += direction * yaw_offset
    else:
        # 低头 (修改 Pitch, index=4)
        direction = np.random.choice([-1, 1])
        pitch_offset = np.random.uniform(*PITCH_DISTRACTION_RANGE)
        features_copy[:, 4] += direction * pitch_offset

    return features_copy


def classify_drowsiness_level(features):
    """
    根据平均 EAR 值分类困倦程度
    返回: 'light_drowsy' 或 'severe_drowsy'
    """
    avg_ear = np.mean(features[:, 0])  # EAR 是第一个特征

    if avg_ear < EAR_SEVERE_DROWSY_THRESHOLD:
        return "severe_drowsy"
    elif avg_ear < EAR_LIGHT_DROWSY_THRESHOLD:
        return "light_drowsy"
    else:
        # EAR 过高可能是误标记,记录警告但仍归类
        print(f"  警告: EAR={avg_ear:.3f} 高于阈值,可能是误标记数据")
        return "light_drowsy"


def process_video_segment(image_paths, prefix, category, face_mesh):
    """处理一个视频片段"""
    # 按帧号排序
    image_paths.sort(key=lambda x: x[0])

    features_list = []

    # 处理每一帧
    valid_frames = 0
    for frame_num, img_path in image_paths:
        try:
            img = imread_unicode(img_path)
        except:
            continue

        # 提取特征
        features, landmarks = calculate_all_features(img, face_mesh)

        if features is not None:
            features_list.append(features)
            valid_frames += 1

    # 调试信息
    if valid_frames == 0:
        print(f"警告: {category}/{prefix} - 0/{len(image_paths)} 帧检测到人脸")
    elif valid_frames < len(image_paths) * 0.5:
        print(
            f"警告: {category}/{prefix} - 仅 {valid_frames}/{len(image_paths)} 帧检测到人脸"
        )

    # 转换为numpy数组
    if len(features_list) > 0:
        return np.array(features_list)
    else:
        return None


def augment_features(features, aug_type="noise"):
    """
    数据增强: 生成变化的特征序列

    Args:
        features: (num_frames, 7) 特征数组
        aug_type: 增强类型 ['noise', 'scale', 'shift', 'mixed']

    Returns:
        增强后的特征
    """
    features_aug = features.copy()

    if aug_type == "noise":
        # 添加高斯噪声
        noise = np.random.normal(0, 0.02, features.shape)
        features_aug += noise

    elif aug_type == "scale":
        # 特征缩放
        scale_factors = np.random.uniform(0.95, 1.05, features.shape[1])
        features_aug *= scale_factors

    elif aug_type == "shift":
        # 时间偏移(删除前几帧或后几帧)
        shift = np.random.randint(-5, 5)
        if shift > 0:
            features_aug = features[shift:]
        elif shift < 0:
            features_aug = features[:shift]

    elif aug_type == "mixed":
        # 混合增强
        noise = np.random.normal(0, 0.01, features.shape)
        scale = np.random.uniform(0.98, 1.02, features.shape[1])
        features_aug = features * scale + noise

    return features_aug


def oversample_minority_classes(stats, groups, category, face_mesh, target_count=None):
    """
    对少数类进行过采样

    Args:
        stats: 统计字典
        groups: 视频片段分组
        category: 类别名称
        face_mesh: MediaPipe face mesh
        target_count: 目标样本数(None则使用最大类别数量)
    """
    if target_count is None:
        # 使用当前最大的类别数作为目标
        target_count = max(
            stats["awake"],
            stats["light_drowsy"],
            stats["severe_drowsy"],
            stats["distracted"],
        )

    current_counts = {
        "awake": stats["awake"],
        "light_drowsy": stats["light_drowsy"],
        "severe_drowsy": stats["severe_drowsy"],
        "distracted": stats["distracted"],
    }

    # 对每个少数类进行过采样
    for class_name, current_count in current_counts.items():
        if current_count < target_count and current_count > 0:
            shortage = target_count - current_count
            print(f"\n对 {class_name} 进行过采样: 需要增加 {shortage} 个样本")

            # 找到该类别的所有现有文件
            existing_files = [
                f
                for f in os.listdir(OUTPUT_NPY_DIR)
                if f.startswith(class_name) and f.endswith(".npy")
            ]

            if len(existing_files) == 0:
                continue

            # 循环增强直到达到目标数量
            aug_count = 0
            aug_types = ["noise", "scale", "mixed"]

            while aug_count < shortage:
                # 随机选择一个现有文件
                source_file = np.random.choice(existing_files)
                features = np.load(os.path.join(OUTPUT_NPY_DIR, source_file))

                # 随机选择增强方式
                aug_type = np.random.choice(aug_types)
                augmented = augment_features(features, aug_type)

                # 保存增强样本
                aug_filename = f"{class_name}_aug{aug_count}.npy"
                np.save(os.path.join(OUTPUT_NPY_DIR, aug_filename), augmented)

                stats[class_name] += 1
                stats["total_frames"] += len(augmented)
                aug_count += 1


def process_all_videos():
    """处理所有视频片段"""
    os.makedirs(OUTPUT_NPY_DIR, exist_ok=True)

    # 调试:检查数据集路径
    print(f"\n=== 调试信息 ===")
    print(f"数据集路径: {DATASET_PATH}")
    print(f"路径是否存在: {os.path.exists(DATASET_PATH)}")

    if os.path.exists(DATASET_PATH):
        print(f"数据集下的文件夹: {os.listdir(DATASET_PATH)}")

    # 统计信息
    stats = {
        "awake": 0,
        "light_drowsy": 0,
        "severe_drowsy": 0,
        "distracted": 0,
        "total_frames": 0,
    }

    # 修复: 使用 mp_face_mesh.FaceMesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:

        # 第一步：处理 Drowsy 数据
        category = "Drowsy"
        folder_path = os.path.join(DATASET_PATH, category)
        print(f"\n=== 处理 {category} 数据 ===")

        if os.path.isdir(folder_path):
            all_files = [
                f
                for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ]

            groups = defaultdict(list)
            for fname in all_files:
                match = re.match(
                    r"^([a-zA-Z]+)(\d+)\.(jpg|png|jpeg)$", fname, re.IGNORECASE
                )
                if match:
                    prefix = match.group(1).upper()
                    frame_num = int(match.group(2))
                    img_path = os.path.join(folder_path, fname)
                    groups[prefix].append((frame_num, img_path))

            for prefix, image_paths in tqdm(groups.items(), desc="Processing Drowsy"):
                features = process_video_segment(
                    image_paths, prefix, category, face_mesh
                )

                if features is not None and len(features) > 0:
                    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                        print(f"跳过 {prefix} - 包含无效值(NaN/Inf)")
                        continue

                    drowsy_level = classify_drowsiness_level(features)

                    # 保存原始样本
                    npy_path = os.path.join(
                        OUTPUT_NPY_DIR, f"{drowsy_level}_{prefix}.npy"
                    )
                    np.save(npy_path, features)
                    stats[drowsy_level] += 1
                    stats["total_frames"] += len(features)

                    # 数据增强: 生成额外样本
                    if ENABLE_AUGMENTATION:
                        for i in range(AUGMENTATION_RATIO):
                            aug_type = np.random.choice(["noise", "scale", "mixed"])
                            augmented = augment_features(features, aug_type)
                            aug_path = os.path.join(
                                OUTPUT_NPY_DIR, f"{drowsy_level}_{prefix}_aug{i}.npy"
                            )
                            np.save(aug_path, augmented)
                            stats[drowsy_level] += 1
                            stats["total_frames"] += len(augmented)

        # 第二步：处理 Non Drowsy 数据
        category = "Non Drowsy"
        folder_path = os.path.join(DATASET_PATH, category)
        print(f"\n=== 处理 {category} 数据 ===")

        if os.path.isdir(folder_path):
            all_files = [
                f
                for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ]

            groups = defaultdict(list)
            for fname in all_files:
                match = re.match(
                    r"^([a-zA-Z]+)(\d+)\.(jpg|png|jpeg)$", fname, re.IGNORECASE
                )
                if match:
                    prefix = match.group(1).upper()
                    frame_num = int(match.group(2))
                    img_path = os.path.join(folder_path, fname)
                    groups[prefix].append((frame_num, img_path))

            prefixes = list(groups.keys())
            np.random.shuffle(prefixes)

            num_distracted = int(len(prefixes) * DISTRACTION_RATIO)
            distracted_prefixes = set(prefixes[:num_distracted])

            for prefix, image_paths in tqdm(
                groups.items(), desc="Processing Non Drowsy"
            ):
                features = process_video_segment(
                    image_paths, prefix, category, face_mesh
                )

                if features is not None and len(features) > 0:
                    if prefix in distracted_prefixes:
                        distracted_features = synthesize_distraction(features)
                        npy_path = os.path.join(
                            OUTPUT_NPY_DIR, f"distracted_{prefix}.npy"
                        )
                        np.save(npy_path, distracted_features)
                        stats["distracted"] += 1
                        stats["total_frames"] += len(distracted_features)

                        # 增强分心数据
                        if ENABLE_AUGMENTATION:
                            for i in range(AUGMENTATION_RATIO):
                                aug_features = augment_features(
                                    distracted_features, "mixed"
                                )
                                aug_path = os.path.join(
                                    OUTPUT_NPY_DIR, f"distracted_{prefix}_aug{i}.npy"
                                )
                                np.save(aug_path, aug_features)
                                stats["distracted"] += 1
                                stats["total_frames"] += len(aug_features)
                    else:
                        npy_path = os.path.join(OUTPUT_NPY_DIR, f"awake_{prefix}.npy")
                        np.save(npy_path, features)
                        stats["awake"] += 1
                        stats["total_frames"] += len(features)

                        # 增强清醒数据
                        if ENABLE_AUGMENTATION:
                            for i in range(AUGMENTATION_RATIO):
                                aug_features = augment_features(features, "noise")
                                aug_path = os.path.join(
                                    OUTPUT_NPY_DIR, f"awake_{prefix}_aug{i}.npy"
                                )
                                np.save(aug_path, aug_features)
                                stats["awake"] += 1
                                stats["total_frames"] += len(aug_features)

        # 第三步：平衡数据集(过采样少数类)
        print("\n=== 平衡数据集 ===")
        oversample_minority_classes(stats, groups, category, face_mesh)

    print("\n=== 处理完成 ===")
    print(f"清醒 (Awake): {stats['awake']}")
    print(f"轻度困倦 (Light Drowsy): {stats['light_drowsy']}")
    print(f"重度困倦 (Severe Drowsy): {stats['severe_drowsy']}")
    print(f"分心 (Distracted): {stats['distracted']}")
    print(
        f"总样本数: {sum([stats['awake'], stats['light_drowsy'], stats['severe_drowsy'], stats['distracted']])}"
    )
    print(f"总帧数: {stats['total_frames']}")
    print(f"特征保存路径: {OUTPUT_NPY_DIR}")


if __name__ == "__main__":
    process_all_videos()
