import cv2
import numpy as np
import mediapipe as mp

mp_facemesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates

# 眼睛关键点索引
chosen_left_eye_idxs = [362, 385, 387, 263, 373, 380]
chosen_right_eye_idxs = [33, 160, 158, 133, 153, 144]


def distance(point_1, point_2):
    """计算两点之间的欧氏距离"""
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist


def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    """
    计算单只眼睛的 Eye Aspect Ratio (EAR)

    Args:
        landmarks: 检测到的关键点列表
        refer_idxs: 选择的关键点索引位置,顺序为 P1, P2, P3, P4, P5, P6
        frame_width: 帧宽度
        frame_height: 帧高度

    Returns:
        ear: Eye Aspect Ratio 值
        coords_points: 关键点坐标列表
    """
    try:
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)

        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

    except:
        ear = 0.0
        coords_points = None

    return ear, coords_points


def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    left_ear, left_lm_coordinates = get_ear(landmarks, left_eye_idxs, image_w, image_h)
    right_ear, right_lm_coordinates = get_ear(
        landmarks, right_eye_idxs, image_w, image_h
    )
    avg_ear = (left_ear + right_ear) / 2.0

    return avg_ear, left_ear, right_ear, (left_lm_coordinates, right_lm_coordinates)


def draw_eye_landmarks(frame, coords_points, color=(0, 255, 0)):
    if coords_points:
        for coord in coords_points:
            if coord:
                cv2.circle(frame, coord, 2, color, -1)


def main():
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

            status_text = "Status: No face detected"
            text_color = (0, 255, 255)  # 黄色

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark

                    avg_ear, left_ear, right_ear, (left_coords, right_coords) = (
                        calculate_avg_ear(
                            landmarks,
                            chosen_left_eye_idxs,
                            chosen_right_eye_idxs,
                            imgW,
                            imgH,
                        )
                    )

                    # 绘制眼睛关键点
                    draw_eye_landmarks(frame, left_coords, (255, 0, 0))
                    draw_eye_landmarks(frame, right_coords, (255, 0, 0))

                    # 显示 EAR 值
                    status_text = f"left eye EAR: {left_ear:.3f} | right eye EAR: {right_ear:.3f} | average EAR: {avg_ear:.3f}"
                    text_color = (0, 255, 0)  # 绿色

                    if left_coords and left_coords[0]:
                        cv2.putText(
                            frame,
                            f"L: {left_ear:.3f}",
                            (left_coords[0][0] - 30, left_coords[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 0, 0),
                            2,
                        )

                    if right_coords and right_coords[0]:
                        cv2.putText(
                            frame,
                            f"R: {right_ear:.3f}",
                            (right_coords[0][0] - 30, right_coords[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 0, 0),
                            2,
                        )

            cv2.putText(
                frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                text_color,
                2,
            )

            cv2.imshow("Eye Aspect Ratio Detection", frame)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
