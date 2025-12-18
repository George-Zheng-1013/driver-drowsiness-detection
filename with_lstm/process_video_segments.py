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
EAR_LIGHT_DROWSY_THRESHOLD = 0.25
EAR_SEVERE_DROWSY_THRESHOLD = 0.15

# 眨眼过滤阈值（用于 Non Drowsy 数据）
EAR_BLINK_THRESHOLD = 0.2  # 低于此值视为眨眼，将被丢弃

# 分心合成参数
DISTRACTION_RATIO = 0.3
YAW_DISTRACTION_RANGE = (30, 60)
PITCH_DISTRACTION_RANGE = (15, 25)

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
    合成分心数据：模拟真实的分心场景（主要是左右转头看后视镜、看窗外等）

    features: [EAR, MAR, Circ, Brow, Pitch, Yaw, Roll]

    核心思路:
    1. 分心主要表现为左右转头 (Yaw 变化为主)
    2. 侧脸时，MediaPipe 检测眼睛会出现异常（EAR 波动、Circularity 下降）
    3. 分心时通常不会打哈欠，MAR 保持低值
    4. 可能伴随轻微的头部倾斜 (Roll)
    """
    features_copy = features.copy()

    # === 1. 主要动作：左右转头 (修改 Yaw) ===
    # 80% 是纯左右转头，20% 是转头+轻微低头(看中控台)
    distraction_scenario = np.random.random()

    if distraction_scenario < 0.8:
        # 场景1: 纯左右转头 (看后视镜/侧窗)
        direction = np.random.choice([-1, 1])  # -1左转, +1右转
        yaw_offset = np.random.uniform(30, 60)  # 转头角度
        features_copy[:, 5] += direction * yaw_offset  # Yaw (index=5)

        # 可能伴随轻微的头部倾斜
        if np.random.random() < 0.3:
            roll_offset = np.random.uniform(5, 15) * direction
            features_copy[:, 6] += roll_offset  # Roll (index=6)

        distraction_intensity = abs(yaw_offset) / 60.0  # 归一化强度

    else:
        # 场景2: 转头看中控台 (Yaw + 轻微 Pitch)
        yaw_offset = np.random.uniform(20, 40) * np.random.choice([-1, 1])
        pitch_offset = np.random.uniform(5, 12)  # 轻微低头
        features_copy[:, 5] += yaw_offset  # Yaw
        features_copy[:, 4] += pitch_offset  # Pitch (index=4)

        distraction_intensity = 0.7

    # === 2. 模拟侧脸导致的眼睛检测异常 ===

    # 2.1 EAR 异常 (index=0)
    # 修改点：移除 'dropout'，因为它会让 EAR 看起来像闭眼(Severe Drowsy)
    # 我们只保留 'noise' (噪声) 和 'spike' (异常高值)
    ear_disturbance_strategy = np.random.choice(["noise", "spike"])

    if ear_disturbance_strategy == "noise":
        # 策略A: 添加噪声，但确保 EAR 保持在"睁眼"范围
        noise_std = 0.04 * distraction_intensity
        ear_noise = np.random.normal(0, noise_std, features_copy.shape[0])
        features_copy[:, 0] += ear_noise
        # 关键修改：下限提高到 0.20，绝对不能低于 0.18 (Severe阈值是0.15)
        features_copy[:, 0] = np.clip(features_copy[:, 0], 0.20, 0.38)

    elif ear_disturbance_strategy == "spike":
        # 策略B: 随机帧出现尖峰 (检测失效通常会导致数值乱跳，而不是一直闭眼)
        num_frames = features_copy.shape[0]
        num_spikes = int(num_frames * np.random.uniform(0.1, 0.3))
        spike_indices = np.random.choice(num_frames, num_spikes, replace=False)

        for idx in spike_indices:
            # 设为异常值
            if np.random.random() < 0.5:
                features_copy[idx, 0] = np.random.uniform(0.35, 0.45)  # 异常大
            else:
                features_copy[idx, 0] = np.random.uniform(0.18, 0.22)  # 偏小但不是闭眼

    # 2.2 Circularity (index=2)
    # 稍微降低一点，但不要降太多
    circ_reduction = np.random.uniform(0.05, 0.15) * distraction_intensity
    features_copy[:, 2] -= circ_reduction
    features_copy[:, 2] = np.clip(features_copy[:, 2], 0.4, 1.0)  # 提高下限

    # 2.3 眉毛距离异常（侧脸时眉毛关键点漂移）
    brow_noise = np.random.normal(
        0, 0.025 * distraction_intensity, features_copy.shape[0]
    )
    features_copy[:, 3] += brow_noise  # Brow Distance (index=3)
    features_copy[:, 3] = np.clip(features_copy[:, 3], 0.1, 0.5)

    # === 3. MAR 保持低值（分心时不会打哈欠）===
    # 强制压制 MAR，避免被误判为疲劳
    features_copy[:, 1] = np.minimum(features_copy[:, 1], 0.3)  # MAR (index=1)
    # 添加小幅噪声保持自然性
    mar_noise = np.random.normal(0, 0.015, features_copy.shape[0])
    features_copy[:, 1] += mar_noise
    features_copy[:, 1] = np.clip(features_copy[:, 1], 0.0, 0.3)

    # === 4. 添加时序变化（模拟转头过程）===
    # 30% 概率模拟"快速转头"的突变
    if np.random.random() < 0.3:
        # 在中间某一帧快速转头
        transition_frame = np.random.randint(
            int(features_copy.shape[0] * 0.3), int(features_copy.shape[0] * 0.7)
        )

        # 转头前后 Yaw 有明显差异
        yaw_jump = np.random.uniform(10, 20) * np.random.choice([-1, 1])
        features_copy[transition_frame:, 5] += yaw_jump

        # 转头瞬间 EAR 可能出现异常
        if transition_frame > 0:
            features_copy[
                transition_frame - 1 : transition_frame + 2, 0
            ] *= np.random.uniform(0.8, 1.2)
            features_copy[:, 0] = np.clip(features_copy[:, 0], 0.1, 0.4)

    # === 5. 避免 Pitch 出现大幅单纯的上下变化 ===
    # 如果有 Pitch 变化，必须伴随 Yaw (符合真实转头逻辑)
    # 这一步已经在场景2中处理了，这里不需要额外操作

    return features_copy


# 新增阈值配置
MAR_YAWN_THRESHOLD = 0.5  # 打哈欠阈值


def classify_drowsiness_level(features):
    """
    根据 EAR 和 MAR 分类困倦程度
    features: [EAR, MAR, Circ, Brow, Pitch, Yaw, Roll]
    """
    avg_ear = features[:, 0].mean()  # EAR 是第1个特征
    avg_mar = features[:, 1].mean()  # MAR 是第2个特征

    # 1. 严重疲劳: 眼睛闭合严重
    if avg_ear < EAR_SEVERE_DROWSY_THRESHOLD:
        return "severe_drowsy"

    # 2. 轻度疲劳: 眼睛半眯 OR 正在打哈欠
    # 逻辑: EAR在中间范围，或者 EAR正常但嘴巴张大(打哈欠)
    elif avg_ear < EAR_LIGHT_DROWSY_THRESHOLD or avg_mar > MAR_YAWN_THRESHOLD:
        return "light_drowsy"

    # 3. 既不闭眼也不打哈欠 -> 视为该片段实际上是清醒的(即使在Drowsy文件夹)
    else:
        # 返回 None 表示这是一个"脏数据"，应该丢弃，不要强行标记为 light_drowsy
        print(f"  [过滤] EAR={avg_ear:.3f}, MAR={avg_mar:.3f} -> 看起来太清醒，丢弃")
        return None


def process_video_segment(
    image_paths, prefix, category, face_mesh, filter_blinks=False
):
    """
    处理一个视频片段

    Args:
        image_paths: 图片路径列表
        prefix: 视频前缀
        category: 类别名称
        face_mesh: MediaPipe face mesh
        filter_blinks: 是否过滤眨眼帧（仅用于 Non Drowsy 数据）
    """
    # 按帧号排序
    image_paths.sort(key=lambda x: x[0])

    features_list = []

    # 处理每一帧
    valid_frames = 0
    blink_filtered_frames = 0

    for frame_num, img_path in image_paths:
        try:
            img = imread_unicode(img_path)
        except:
            continue

        # 提取特征
        features, landmarks = calculate_all_features(img, face_mesh)

        if features is not None:
            # 如果启用眨眼过滤，检查 EAR
            if filter_blinks:
                ear = features[0]  # EAR 是第一个特征
                if ear < EAR_BLINK_THRESHOLD:
                    blink_filtered_frames += 1
                    continue  # 跳过眨眼帧

            features_list.append(features)
            valid_frames += 1

    # 调试信息
    if filter_blinks and blink_filtered_frames > 0:
        print(f"  {category}/{prefix} - 过滤了 {blink_filtered_frames} 帧眨眼")

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
        "blink_filtered": 0,  # 新增：记录过滤的眨眼帧数
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
                    image_paths,
                    prefix,
                    category,
                    face_mesh,
                    filter_blinks=False,  # Drowsy 不过滤
                )

                if features is not None and len(features) > 0:
                    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                        print(f"跳过 {prefix} - 包含无效值(NaN/Inf)")
                        continue

                    drowsy_level = classify_drowsiness_level(features)

                    # === 修改开始: 处理 None 返回值 ===
                    if drowsy_level is None:
                        continue  # 跳过这个看起来像清醒的样本
                    # === 修改结束 ===

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
                # 关键修改：启用眨眼过滤
                features = process_video_segment(
                    image_paths, prefix, category, face_mesh, filter_blinks=True
                )

                if features is not None and len(features) > 0:
                    # 再次检查：过滤后如果平均 EAR 仍然过低，说明整个片段质量不佳
                    avg_ear = features[:, 0].mean()
                    if avg_ear < EAR_BLINK_THRESHOLD:
                        print(
                            f"  [丢弃] {prefix} - 过滤后平均 EAR={avg_ear:.3f} 仍过低"
                        )
                        continue

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
