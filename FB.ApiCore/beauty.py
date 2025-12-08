from typing import Tuple

import cv2
import mediapipe as mp
import numpy as np
from sklearn.cluster import KMeans


class HybridSkinRetouch:
    """ML-based acne detection + frequency separation healing"""

    def __init__(self, n_clusters: int = 5) -> None:
        self.k = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
        )

    def get_face_region_mask(self, img: np.ndarray) -> np.ndarray | None:
        """Tạo mask vùng mặt bằng MediaPipe để chỉ xử lý trong khuôn mặt"""
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        # Lấy toàn bộ khuôn mặt (không loại bỏ mắt/môi)
        face_oval = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in face_oval])
        cv2.fillPoly(mask, [pts], 255)

        # Làm mềm biên mask
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        return mask

    def get_ml_mask(self, img: np.ndarray, face_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Phát hiện mụn chỉ trong vùng mặt"""
        small_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        small_face_mask = cv2.resize(face_mask, (0, 0), fx=0.5, fy=0.5)
        
        img_lab = cv2.cvtColor(small_img, cv2.COLOR_BGR2LAB)
        
        # Chỉ lấy pixel trong vùng mặt
        face_pixels = img_lab[small_face_mask > 128]
        if len(face_pixels) < 10:
            return np.zeros_like(face_mask), np.zeros_like(face_mask)

        labels_1d = self.kmeans.fit_predict(face_pixels)
        centers = self.kmeans.cluster_centers_

        # Tìm cluster mụn
        scores = centers[:, 1] * 1.5 - centers[:, 0]
        acne_idx = np.argmax(scores)

        # Tạo mask acne
        mask_small = np.zeros(small_img.shape[:2], dtype=np.uint8)
        face_coords = np.where(small_face_mask > 128)
        acne_coords = np.where(labels_1d == acne_idx)[0]
        
        for idx in acne_coords:
            mask_small[face_coords[0][idx], face_coords[1][idx]] = 255

        mask = cv2.resize(mask_small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        mask_soft = cv2.GaussianBlur(mask, (11, 11), 0)
        return mask, mask_soft

    def frequency_separation_heal(self, img: np.ndarray, acne_mask: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
        """Sửa mụn tự nhiên + làm đẹp da trong vùng mặt"""
        # Tách frequency nhẹ hơn để giữ chi tiết
        low_freq = cv2.GaussianBlur(img, (15, 15), 0)
        high_freq = cv2.subtract(img, low_freq) + 128

        # LOW FREQ: Làm mịn nhẹ + chỉnh màu da
        healed_low = cv2.bilateralFilter(low_freq, 7, 40, 40)
        
        # Chuyển sang LAB để điều chỉnh màu da
        lab = cv2.cvtColor(healed_low, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Tăng sáng nhẹ + giảm đỏ một chút
        l = cv2.add(l, 8)
        a = cv2.add(a, -3)
        
        healed_low = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        # Chỉ áp dụng vào vùng mụn trong mặt
        acne_f = acne_mask.astype(float) / 255.0
        acne_3ch = np.stack([acne_f] * 3, axis=-1)

        final_low = (healed_low * acne_3ch + low_freq * (1 - acne_3ch))

        # HIGH FREQ: Giữ nhiều texture hơn để da tự nhiên
        flat_gray = np.full_like(high_freq, 128)
        healed_high = cv2.addWeighted(high_freq, 0.5, flat_gray, 0.5, 0)
        final_high = (healed_high * acne_3ch + high_freq * (1 - acne_3ch))

        # Gộp lại
        healed = cv2.addWeighted(final_low.astype(np.uint8), 1.0, final_high.astype(np.uint8), 1.0, -128)

        # Thêm hạt nhiễu nhẹ để da không bị nhựa
        h_img, w_img, c = healed.shape
        noise = np.random.randn(h_img, w_img, c) * 5
        healed = np.clip(healed.astype(float) + noise, 0, 255).astype(np.uint8)

        # Blend thêm 20% ảnh gốc để giữ tự nhiên
        face_area = (face_mask > 50).astype(np.uint8)
        face_3ch = np.stack([face_area] * 3, axis=-1)
        
        healed_in_face = cv2.addWeighted(healed, 0.8, img, 0.2, 0)
        result = (healed_in_face * face_3ch + img * (1 - face_3ch)).astype(np.uint8)
        
        return result

    def run(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
        if img is None:
            return None, None

        h, w = img.shape[:2]
        if w > 1000:
            scale = 1000 / w
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

        # Bước 1: Lấy mask mặt
        face_mask = self.get_face_region_mask(img)
        if face_mask is None:
            return None, None

        # Bước 2: Phát hiện mụn trong vùng mặt
        _, acne_mask_soft = self.get_ml_mask(img, face_mask)

        # Bước 3: Chữa mụn chỉ trong mặt
        result = self.frequency_separation_heal(img, acne_mask_soft, face_mask)

        return img, result


class UltimateBeautyCam:
    def __init__(self) -> None:
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
        )

    def get_face_mask(self, img: np.ndarray) -> np.ndarray | None:
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        face_oval = [
            10,
            338,
            297,
            332,
            284,
            251,
            389,
            356,
            454,
            323,
            361,
            288,
            397,
            365,
            379,
            378,
            400,
            377,
            152,
            148,
            176,
            149,
            150,
            136,
            172,
            58,
            132,
            93,
            234,
            127,
            162,
            21,
            54,
            103,
            67,
            109,
        ]
        pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in face_oval])
        cv2.fillPoly(mask, [pts], 255)

        eyes_lips = [
            [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 267],
            [55, 65, 52, 53, 46],
            [285, 295, 282, 283, 276],
        ]
        for parts in eyes_lips:
            pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in parts])
            cv2.fillPoly(mask, [pts], 0)

        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        return mask

    def skin_smoothing_strong(self, img: np.ndarray) -> np.ndarray:
        smooth = cv2.edgePreservingFilter(img, flags=1, sigma_s=50, sigma_r=0.4)
        smooth = cv2.edgePreservingFilter(smooth, flags=1, sigma_s=40, sigma_r=0.3)
        return smooth

    def add_grain(self, img: np.ndarray) -> np.ndarray:
        h, w, c = img.shape
        noise = np.random.randn(h, w, c) * 10
        noisy_img = img.astype(float) + noise
        return np.clip(noisy_img, 0, 255).astype(np.uint8)

    def color_grading(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.add(l, 10)
        a = cv2.add(a, -5)
        merged = cv2.merge((l, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def run(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
        if img is None:
            return None, None

        h, w = img.shape[:2]
        if w > 1200:
            scale = 1200 / w
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

        mask_gray = self.get_face_mask(img)
        if mask_gray is None:
            return None, None

        mask_3ch = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR) / 255.0

        skin_layer = self.skin_smoothing_strong(img)
        skin_layer = self.color_grading(skin_layer)
        skin_layer = self.add_grain(skin_layer)

        final = (skin_layer * mask_3ch + img * (1 - mask_3ch)).astype(np.uint8)
        final = cv2.addWeighted(final, 0.85, img, 0.15, 0)

        return img, final


def decode_image(file_bytes: bytes) -> np.ndarray | None:
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img
