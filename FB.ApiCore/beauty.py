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
        """Tạo mask vùng mặt bằng MediaPipe - LOẠI TRỪ mắt, môi, lông mi"""
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        # Vẽ toàn bộ khuôn mặt
        face_oval = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in face_oval])
        cv2.fillPoly(mask, [pts], 255)

        # Tạo mask exclusion riêng để MỞ RỘNG vùng loại trừ
        exclude_mask = np.zeros((h, w), dtype=np.uint8)
        
        # LOẠI TRỪ mắt trái, mắt phải, môi - GIỮ NGUYÊN MÀU TỰ NHIÊN
        exclude_regions = [
            # Mắt trái FULL (cả trên + dưới + lông mi)
            [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
             130, 25, 110, 24, 23, 22, 26, 112, 243],
            # Mắt phải FULL (cả trên + dưới + lông mi)  
            [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382,
             359, 255, 339, 254, 253, 252, 256, 341, 463],
            # Môi FULL (trên + dưới + viền ngoài)
            [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
             78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
            # Lông mày trái MỞ RỘNG
            [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 124, 35, 111, 117, 118, 119],
            # Lông mày phải MỞ RỘNG
            [300, 293, 334, 296, 336, 285, 295, 282, 283, 276, 353, 265, 340, 346, 347, 348]
        ]
        
        for region in exclude_regions:
            pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in region])
            cv2.fillPoly(exclude_mask, [pts], 255)
        
        # MỞ RỘNG vùng exclusion để chắc chắn không xâm lấn
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        exclude_mask = cv2.dilate(exclude_mask, kernel, iterations=2)
        
        # Làm mềm biên exclusion
        exclude_mask = cv2.GaussianBlur(exclude_mask, (25, 25), 0)
        
        # Trừ vùng exclusion khỏi mask chính
        mask = cv2.subtract(mask, exclude_mask)
        
        # Làm mềm biên mask tổng thể
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
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

        # Tìm NHIỀU cluster mụn (top 2 clusters có màu đỏ/tối nhất)
        scores = centers[:, 1] * 2.0 - centers[:, 0] * 1.2  # Tăng weight cho màu đỏ
        top_acne_indices = np.argsort(scores)[-2:]  # Lấy 2 clusters có score cao nhất

        # Tạo mask acne từ NHIỀU clusters
        mask_small = np.zeros(small_img.shape[:2], dtype=np.uint8)
        face_coords = np.where(small_face_mask > 128)
        
        for acne_idx in top_acne_indices:
            acne_coords = np.where(labels_1d == acne_idx)[0]
            for idx in acne_coords:
                mask_small[face_coords[0][idx], face_coords[1][idx]] = 255

        mask = cv2.resize(mask_small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Mở rộng vùng xử lý để cover hết mụn
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Tăng từ 3x3 lên 5x5
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # CLOSE trước để lấp lỗ
        mask = cv2.dilate(mask, kernel, iterations=2)  # Tăng từ 1 lên 2

        mask_soft = cv2.GaussianBlur(mask, (15, 15), 0)  # Tăng từ 11 lên 15
        return mask, mask_soft

    def frequency_separation_heal(self, img: np.ndarray, acne_mask: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
        """Sửa mụn tự nhiên + làm đẹp da trong vùng mặt"""
        # Tách frequency với blur vừa phải
        low_freq = cv2.GaussianBlur(img, (17, 17), 0)  # Giảm từ 21 xuống 17
        high_freq = cv2.subtract(img, low_freq) + 128

        # LOW FREQ: Làm mịn vừa phải + chỉnh màu da TỰ NHIÊN
        healed_low = cv2.bilateralFilter(low_freq, 7, 50, 50)  # Giảm từ 9,60,60 xuống 7,50,50
        
        # Chuyển sang LAB để điều chỉnh màu da
        lab = cv2.cvtColor(healed_low, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Chỉnh nhẹ để tự nhiên - KHÔNG làm trắng quá
        l = cv2.add(l, 6)   # Giảm từ 12 xuống 6
        a = cv2.add(a, -2)  # Giảm từ -5 lên -2
        
        healed_low = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        # Chỉ áp dụng vào vùng mụn trong mặt
        acne_f = acne_mask.astype(float) / 255.0
        acne_3ch = np.stack([acne_f] * 3, axis=-1)

        final_low = (healed_low * acne_3ch + low_freq * (1 - acne_3ch))

        # HIGH FREQ: Giữ texture tự nhiên hơn
        flat_gray = np.full_like(high_freq, 128)
        healed_high = cv2.addWeighted(high_freq, 0.4, flat_gray, 0.6, 0)  # Tăng từ 0.3 lên 0.4
        final_high = (healed_high * acne_3ch + high_freq * (1 - acne_3ch))

        # Gộp lại
        healed = cv2.addWeighted(final_low.astype(np.uint8), 1.0, final_high.astype(np.uint8), 1.0, -128)

        # Thêm hạt nhiễu để da tự nhiên
        h_img, w_img, c = healed.shape
        noise = np.random.randn(h_img, w_img, c) * 3  # Giảm từ 4 xuống 3
        healed = np.clip(healed.astype(float) + noise, 0, 255).astype(np.uint8)

        # Blend 75% xử lý + 25% gốc để cân bằng tự nhiên
        face_area = (face_mask > 50).astype(np.uint8)
        face_3ch = np.stack([face_area] * 3, axis=-1)
        
        healed_in_face = cv2.addWeighted(healed, 0.75, img, 0.25, 0)  # Giảm từ 0.85 xuống 0.75
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

def decode_image(file_bytes: bytes) -> np.ndarray | None:
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img
