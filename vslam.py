import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R_scipy

def compute_sift_keypoints_descriptors(img):
    sift = cv2.SIFT_create(nfeatures=2000)
    return sift.detectAndCompute(img, None)

def extract_pose_from_images(img1, img2, kp1, des1, kp2, des2, K):
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 10:
        return None, None, None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return None, None, None

    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    return R, t, good_matches

# def draw_trajectory(traj):
#     traj = np.array(traj)
#     plt.figure(figsize=(10, 6))
#     plt.plot(traj[:, 0], traj[:, 2], marker='o', linestyle='-', color='blue', label='Kamera Yolu')
#     plt.title("Kamera Trajektorisi (X-Z Düzlemi)")
#     plt.xlabel("X (metre)")
#     plt.ylabel("Z (metre)")
#     plt.xlim(-10, 10) # X ekseni sınırlarını ayarlayın
#     plt.ylim(-20, 1) # Z ekseni sınırlarını ayarlayın
#     plt.grid(True)
#     plt.axis("equal")
#     plt.legend()
#     plt.show()

def draw_trajectory(traj):
    traj = np.array(traj)
    
    # Z değerlerini ters çevir ve başlangıcı (0,0) yap
    x_coords = traj[:, 0] - traj[0, 0]
    z_coords = -(traj[:, 2] - traj[0, 2])  # Hem merkezleme hem ters çevirme
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, z_coords, marker='o', linestyle='-', color='blue', label='Kamera Yolu')
    
    plt.title("Kamera Trajektorisi (0,0 Sol Altta, Pozitif Metreler)")
    plt.xlabel("X (metre)")
    plt.ylabel("Z (metre)")
    plt.grid(True)
    plt.axis("equal")
    
    # Son noktayı işaretle
    if len(traj) > 0:
        plt.scatter(x_coords[-1], z_coords[-1], color='red', s=100, label='Son Konum')
        plt.text(x_coords[-1], z_coords[-1], f'({x_coords[-1]:.2f}, {z_coords[-1]:.2f})', 
                fontsize=12, ha='right')
    
    plt.legend()
    plt.show()


# ----------------------------
# Ana Akış
# ----------------------------

video_path = "./depo_kisa.mp4" # Video yolunuzu buraya girin
cap = cv2.VideoCapture(video_path)

width, height = 1024, 768
f = width * 0.8
K = np.array([
    [f, 0, width / 2],
    [0, f, height / 2],
    [0, 0, 1]
])

ret, frame_prev = cap.read()
if not ret:
    print("Video açılamadı.")
    exit()

frame_prev = cv2.resize(cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY), (width, height))
kp_prev, des_prev = compute_sift_keypoints_descriptors(frame_prev)

# Başlangıç yönü 180 derece (Y ekseni etrafında) olarak ayarlanıyor.
# Kameranın Z ekseninin başlangıçta negatif X yönüne bakmasını sağlarız.
yaw0_rad = np.deg2rad(0)
R0 = R_scipy.from_euler('xyz', [0, yaw0_rad, 0], degrees=False).as_matrix() # X-Y-Z sırasına göre yaw için Y ekseni dönüşü
# Veya 'zyx' sırasına göre yaw için Z ekseni dönüşü
# R0 = R_scipy.from_euler('zyx', [yaw0_rad, 0, 0], degrees=False).as_matrix()

T_global = np.eye(4)
T_global[:3, :3] = R0 # Global dönüşüm matrisinin rotasyon kısmını başlat

trajectory = [T_global[:3, 3].copy()]
frame_idx = 1
scale = 0.03  # Her kare arası mesafe 10 cm kabul edilir

while True:
    ret, frame_curr_color = cap.read()
    if not ret:
        break

    frame_curr = cv2.resize(cv2.cvtColor(frame_curr_color, cv2.COLOR_BGR2GRAY), (width, height))
    kp_curr, des_curr = compute_sift_keypoints_descriptors(frame_curr)

    R, t, matches = extract_pose_from_images(frame_prev, frame_curr, kp_prev, des_prev, kp_curr, des_curr, K)

    overlay = cv2.resize(frame_curr_color, (width, height))

    if R is not None:
        # Mevcut kareler arası dönüşüm matrisi
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3:] = scale * t  # Ölçekli çeviri

        # Global dönüşümle birleştir
        T_global = T_global @ T
        trajectory.append(T_global[:3, 3].copy())

        # Anlık kareler arası dönüşümün açısını hesapla (dönme miktarı)
        rvec, _ = cv2.Rodrigues(R)
        angle = np.linalg.norm(rvec) * 180 / np.pi

        # Global dönüşüm matrisinden Euler açılarını (Yaw, Pitch, Roll) çıkar
        # 'zyx' sırası genellikle robotikte yaw (Z), pitch (Y), roll (X) olarak kullanılır.
        # Bu, R_global'in sırasına bağlıdır. Eğer 'xyz' veya başka bir sıraya göre tanımlıysa,
        # buradan çıkan değerler farklı eksenleri temsil edebilir.
        R_global_rot = R_scipy.from_matrix(T_global[:3, :3])
        euler_angles_deg = R_global_rot.as_euler('zyx', degrees=True) # ZYX sırasına göre Euler açıları (derece)

        global_yaw = euler_angles_deg[0]  # ZYX için ilk değer Yaw'dır
        global_pitch = euler_angles_deg[1] # İkinci değer Pitch'tir
        global_roll = euler_angles_deg[2]  # Üçüncü değer Roll'dur

        cv2.putText(overlay, f"Frame: {frame_idx}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(overlay, f"Rotation angle: {angle:.2f} deg", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
        cv2.putText(overlay, f"Global Yaw: {global_yaw:.2f} deg", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
        cv2.putText(overlay, f"Global Pitch: {global_pitch:.2f} deg", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
        cv2.putText(overlay, f"Global Roll: {global_roll:.2f} deg", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
    else:
        cv2.putText(overlay, f"Frame: {frame_idx}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(overlay, "Pose Estimation Failed", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Visual SLAM - Angle Tracking", overlay)
    if cv2.waitKey(30) & 0xFF == 27:
        break

    frame_prev, kp_prev, des_prev = frame_curr, kp_curr, des_curr
    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

# --- Trajektori Görselleştir ---
draw_trajectory(trajectory)
