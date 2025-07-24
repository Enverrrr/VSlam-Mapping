import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R_scipy

class SIFT_SLAM:
    def __init__(self, seg_model_path):
        """
        SIFT-tabanlı SLAM sistemini zemin segmentasyonu ile başlatır.
        Şimdi 2D işgal ızgara haritası oluşturma yeteneği eklendi.
        """
        self.seg_model = YOLO(seg_model_path)
        self.sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.04, edgeThreshold=10)
        self.bf = cv2.BFMatcher()

        self.last_frame = None
        self.last_kp = None
        self.last_des = None
        self.trajectory = []
        self.current_pose = np.eye(4) # Global 4x4 poz matrisi (T_World_Camera)
        self.initial_yaw_offset = None # Başlangıç yaw'ını saklamak için
        self.current_yaw_degrees = 0.0 # Anlık yaw değerini saklamak için

        self.point_cloud_map = [] # 3D nokta bulutu haritası (opsiyonel olarak kalabilir)
        
        # Kamera iç parametreleri (Kendi kalibrasyon değerlerinizle değiştirin!)
        self.focal = 800
        self.pp = (320, 240) # Genellikle görüntü genişliği/2, görüntü yüksekliği/2
        self.K = np.array([[self.focal, 0, self.pp[0]],
                           [0, self.focal, self.pp[1]],
                           [0, 0, 1]], dtype=np.float32)

        # --- 2D İşgal Izgara Haritası Parametreleri ---
        self.map_resolution = 0.05 # Harita çözünürlüğü: piksel başına metre (örn: 0.05m/px = 5cm/px)
        self.map_size_meters = 20 # Haritanın dünya koordinatlarındaki genişliği/yüksekliği (metre)
        self.map_pixel_size = int(self.map_size_meters / self.map_resolution) # Haritanın piksel cinsinden boyutu
        
        self.map_origin_x_meters = -self.map_size_meters / 2.0 
        self.map_origin_y_meters = -self.map_size_meters / 2.0 

        # İşgal ızgara haritası: 0=boş/bilinmeyen, 255=dolu/engel
        # Haritayı başlangıçta tamamen "bilinmeyen" olarak başlatmak için 127 de kullanılabilir
        # Ancak cv2.dilate için 0 ve 255 ikili değerleri daha uygundur.
        self.occupancy_grid_map = np.full((self.map_pixel_size, self.map_pixel_size), 0, dtype=np.uint8) 
        # self.occupancy_grid_map = np.full((self.map_pixel_size, self.map_pixel_size), 255, dtype=np.uint8) 

        
        # Robotun tahmini yüksekliği (metre cinsinden, kamera zeminden ne kadar yüksekte)
        self.robot_height = 0.5 

    def process_frame(self, frame):
        # 1. Zemin segmentasyonu
        seg_results = self.seg_model(frame, verbose=False)
        floor_mask = self._get_combined_mask(seg_results, frame.shape)
        
        # 2. SIFT ile mevcut kare için anahtar nokta ve tanımlayıcı tespiti
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)
        
        # frame_count, _update_occupancy_grid içinde kullanılmak üzere
        # dışarıdan çağrıldığında frame_count'ı tutmak için bir mekanizma gerekiyor.
        # Basitlik için burada örnek amaçlı bir sayac ekliyorum.
        # Gerçek uygulamada bunu dışarıdan alabilir veya self.frame_counter olarak saklayabilirsiniz.
        # self.frame_counter = getattr(self, 'frame_counter', 0) + 1 
        # Bunu main döngüsünde zaten yapıyorsunuz, sadece sıkıştırmamak için burada eklemedim.

        if self.last_frame is not None and kp is not None and des is not None and \
           self.last_kp is not None and self.last_des is not None and \
           len(kp) > 20 and len(self.last_kp) > 20:
            
            # 3. Özellik eşleştirme
            matches = self.bf.knnMatch(self.last_des, des, k=2)
            
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            
            if len(good_matches) > 10:
                # 4. Poz tahmini ve 3D nokta kurtarma
                R, t, new_points_3d = self._estimate_pose(good_matches, kp)
                
                if R is not None and t is not None:
                    self._update_trajectory(R, t)
                    
                    # Haritayı 3D noktalarla güncelle (opsiyonel)
                    if new_points_3d is not None and new_points_3d.shape[0] > 0:
                        R_wc = self.current_pose[:3, :3]
                        t_wc = self.current_pose[:3, 3].reshape(3, 1) 
                        global_new_points_3d = (R_wc @ new_points_3d.T).T + t_wc.T 
                        self.point_cloud_map.extend(global_new_points_3d.tolist())

                    # --- 2D İşgal Izgara Haritasını Güncelle ---
                    # Her karede güncelleme yapmak yerine, daha seyrek yapabilirsiniz
                    # if self.frame_counter % 5 == 0: # Eğer SIFT_SLAM içinde frame_counter tutuluyorsa
                    self._update_occupancy_grid(frame, floor_mask, R, t)

        # Güncel frame'i bir sonraki adım için sakla
        self.last_frame = gray
        self.last_kp = kp
        self.last_des = des
        
        return floor_mask, self.current_pose

    def _get_combined_mask(self, results, frame_shape):
        combined_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        if results[0].masks is not None:
            for mask in results[0].masks.data:
                mask_np = mask.cpu().numpy()
                mask_resized = cv2.resize(mask_np.astype('uint8'), 
                                         (frame_shape[1], frame_shape[0]))
                combined_mask = cv2.bitwise_or(combined_mask, mask_resized)
        return combined_mask

    def _estimate_pose(self, matches, current_kp):
        src_pts_np = np.float32([self.last_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts_np = np.float32([current_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        E, mask = cv2.findEssentialMat(dst_pts_np, src_pts_np, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        if E is None or E.shape == (0, 0):
            return None, None, None
        
        if np.sum(mask) < 8: 
            return None, None, None 

        _, R, t, _ = cv2.recoverPose(E, dst_pts_np, src_pts_np, self.K, mask=mask)
        
        if R is None or t is None:
            return None, None, None

        P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1)))) 
        P2 = self.K @ np.hstack((R, t)) 

        src_pts_inliers = src_pts_np[mask.ravel() == 1].reshape(-1, 2).T
        dst_pts_inliers = dst_pts_np[mask.ravel() == 1].reshape(-1, 2).T

        if src_pts_inliers.shape[1] == 0:
            return R, t, None 

        triangulated_homogeneous_points = cv2.triangulatePoints(P1, P2, src_pts_inliers, dst_pts_inliers)
        
        if triangulated_homogeneous_points.shape[1] == 0:
            return R, t, None

        triangulated_homogeneous_points = triangulated_homogeneous_points.T 

        valid_indices = triangulated_homogeneous_points[:, 3] > 1e-6 
        triangulated_points_3d = triangulated_homogeneous_points[valid_indices, :3] / triangulated_homogeneous_points[valid_indices, 3:]

        valid_depth_indices = (triangulated_points_3d[:, 2] > 0.1) & (triangulated_points_3d[:, 2] < 100)
        triangulated_points_3d = triangulated_points_3d[valid_depth_indices] 

        if triangulated_points_3d.shape[0] == 0:
            return R, t, None

        return R, t, triangulated_points_3d

    def _update_trajectory(self, R, t):
        scale = 0.03 
        
        T_prev_curr = np.eye(4)
        T_prev_curr[:3, :3] = R
        T_prev_curr[:3, 3] = t.ravel() * scale
        
        self.current_pose = self.current_pose @ T_prev_curr 
        
        self.trajectory.append(self.current_pose[:3, 3].copy())
        if T_prev_curr is not None:
            self.current_pose = self.current_pose @ T_prev_curr

            # Rotation matrisini al
            rotation_matrix = self.current_pose[:3, :3]

            # Rotation matrisini Euler açılarına (rad_roll, rad_pitch, rad_yaw) çevir
            # 'xyz' sırası, dönme sırasını belirtir (x ekseni etrafında roll, y ekseni etrafında pitch, z ekseni etrafında yaw)
            # Eğer farklı bir dönme sırası kullanılıyorsa, burayı değiştirmek gerekebilir.
            r_scipy = R_scipy.from_matrix(rotation_matrix)
            euler_angles_rad = r_scipy.as_euler('xyz', degrees=False) # Radyan cinsinden al

            # Yaw (z ekseni etrafındaki dönme) değerini al
            current_yaw_rad = euler_angles_rad[2] # euler_angles_rad[0]=roll, [1]=pitch, [2]=yaw

            # İlk karede başlangıç yaw'ını kaydet
            if self.initial_yaw_offset is None:
                self.initial_yaw_offset = current_yaw_rad

            # Başlangıç yaw'ını 0'a sabitlemek için offset uygula
            adjusted_yaw_rad = current_yaw_rad - self.initial_yaw_offset
            self.current_yaw_degrees = np.degrees(adjusted_yaw_rad) # Dereceye çevir


    def _update_occupancy_grid(self, frame, floor_mask, R_relative, t_relative):
        robot_x_world = self.current_pose[0, 3]
        robot_y_world = self.current_pose[1, 3]
        
        print(f"DEBUG MAP: Robot World Pos: X={robot_x_world:.2f}, Y={robot_y_world:.2f}")

        robot_map_x, robot_map_y = self._world_to_map(robot_x_world, robot_y_world)
        print(f"DEBUG MAP: Robot Map Pos: X={robot_map_x}, Y={robot_map_y}")

        if 0 <= robot_map_x < self.map_pixel_size and 0 <= robot_map_y < self.map_pixel_size:
            print("DEBUG MAP: Robot harita sınırları içinde.")
            rows, cols = np.mgrid[max(0, robot_map_y - 2):min(self.map_pixel_size, robot_map_y + 3),
                                  max(0, robot_map_x - 2):min(self.map_pixel_size, robot_map_x + 3)]
            self.occupancy_grid_map[rows, cols] = 255 # Boş olarak işaretle
            # print(f"DEBUG MAP: {len(rows.flatten())} hücre boş olarak işaretlendi.")
            # Harita genelinde kaç hücre 0, kaç hücre 255 oldu?
            # print(f"DEBUG MAP: Haritada boş (0) hücre sayısı: {np.sum(self.occupancy_grid_map == 0)}")
            # print(f"DEBUG MAP: Haritada dolu (255) hücre sayısı: {np.sum(self.occupancy_grid_map == 255)}")

        else:
            print("DEBUG MAP: Robot harita sınırları dışında! Harita güncellenmiyor.")
            print(f"           Map_X: {robot_map_x}, Map_Y: {robot_map_y}")

        # --- ESKİ ENGEL İŞLEME KODU BURADAN SONRA GELİYORDU VE ŞİMDİLİK YORUM SATIRI YAPILIYOR ---
        # obstacle_mask = (floor_mask == 0).astype(np.uint8) 
        # obstacle_pixels_y, obstacle_pixels_x = np.where(obstacle_mask == 1)
        # if len(obstacle_pixels_x) == 0: return 
        # cx, cy = self.K[0,2], self.K[1,2]
        # fx, fy = self.K[0,0], self.K[1,1]
        # x_norm = (obstacle_pixels_x - cx) / fx
        # y_norm = (obstacle_pixels_y - cy) / fy
        # Z_obstacle_guess = 1.5 # metre 
        # points_cam = np.vstack((x_norm * Z_obstacle_guess, 
        #                         y_norm * Z_obstacle_guess, 
        #                         np.full_like(x_norm, Z_obstacle_guess)))
        # R_wc = self.current_pose[:3, :3]
        # t_wc = self.current_pose[:3, 3].reshape(3, 1) 
        # points_world = (R_wc @ points_cam) + t_wc
        # world_ox = points_world[0, :]
        # world_oy = points_world[1, :]
        # map_ox_pixels = ((world_ox - self.map_origin_x_meters) / self.map_resolution).astype(int)
        # map_oy_pixels = ((world_oy - self.map_origin_y_meters) / self.map_resolution).astype(int)
        # valid_map_indices = (map_ox_pixels >= 0) & (map_ox_pixels < self.map_pixel_size) & \
        #                     (map_oy_pixels >= 0) & (map_oy_pixels < self.map_pixel_size)
        # valid_map_ox = map_ox_pixels[valid_map_indices]
        # valid_map_oy = map_oy_pixels[valid_map_indices]
        # if len(valid_map_ox) > 0:
        #     self.occupancy_grid_map[valid_map_oy, valid_map_ox] = 255 
        #     kernel = np.ones((3,3),np.uint8) 
        #     self.occupancy_grid_map = cv2.dilate(self.occupancy_grid_map, kernel, iterations = 1)


    def _world_to_map(self, world_x, world_y):
        # Eğer harita eksenleri tersse, burayı değiştir:
        # map_x = int((world_x - self.map_origin_x_meters) / self.map_resolution)
        # map_y = int((world_y - self.map_origin_y_meters) / self.map_resolution)

        # Olası denemeler:
        # 1. Y eksenini ters çevir (eğer haritada yukarı gitmesi gereken aşağı gidiyorsa)
        map_x = int((-world_x - self.map_origin_x_meters) / self.map_resolution)
        map_y = int((-world_y - self.map_origin_y_meters) / self.map_resolution) # Y'yi ters çevir

        # 2. X ve Y'yi takas et (eğer harita 90 derece dönmüş gibiyse)
        # map_x = int((world_y - self.map_origin_x_meters) / self.map_resolution)
        # map_y = int((world_x - self.map_origin_y_meters) / self.map_resolution)

        # 3. Hem takas hem ters çevir
        # map_x = int((world_y - self.map_origin_y_meters) / self.map_resolution) # Y dünya koordinatını X haritaya
        # map_y = int((-world_x - self.map_origin_x_meters) / self.map_resolution) # -X dünya koordinatını Y haritaya
        # map_y = int((self.map_size_meters - (world_x - self.map_origin_x_meters)) / self.map_resolution) # veya haritanın Y eksenini baştan başlatarak ters çevir

        # Bu denemeleri outputa bakarak yapmalısın.
        # Şimdilik orijinal halini koru ve aşağıdaki 3. adımdaki _update_trajectory'yi kontrol edelim
        # map_x = int((world_x - self.map_origin_x_meters) / self.map_resolution)
        # map_y = int((world_y - self.map_origin_y_meters) / self.map_resolution)
        return map_x, map_y

    def _map_to_world(self, map_x, map_y):
        """
        Harita (piksel) koordinatlarını dünya koordinatlarına dönüştürür.
        """
        world_x = map_x * self.map_resolution + self.map_origin_x_meters
        world_y = map_y * self.map_resolution + self.map_origin_y_meters
        return world_x, world_y

    def visualize(self, frame, floor_mask):
        """
        Giriş karesini segmentasyon maskesi ve pozisyon bilgisiyle görselleştirir.
        Ayrıca, 2D işgal ızgara haritasını da gösterir.
        """
        vis_frame = cv2.addWeighted(frame, 0.7, 
                              cv2.cvtColor(floor_mask * 255, cv2.COLOR_GRAY2BGR), 0.3, 0)
        
        x, y, z = self.current_pose[:3, 3]
        cv2.putText(vis_frame, f"Position: X={x:.2f}, Y={y:.2f}, Z={z:.2f}", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Yaw: {self.current_yaw_degrees:.2f} degrees",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # Yaw bilgisini ekle
        
        # --- 2D İşgal Izgara Haritasını Görselleştir ---
        map_display = np.zeros((self.map_pixel_size, self.map_pixel_size, 3), dtype=np.uint8)
        
        # Harita değerlerini görselleştirme için renklere dönüştür
        # `cv2.dilate` kullanıldığı için haritada 0 (boş) ve 255 (dolu) değerleri olacaktır.
        # Bilinmeyen alanları gri olarak göstermek istiyorsan, bu kısmı ayırman gerekebilir
        # Örneğin, başlangıçta 127 ile doldurup, sonra 0 (boş) veya 255 (dolu) olarak güncelleyebilirsin.
        
        # Haritayı BGR formatına dönüştürüp renkler
        map_display[self.occupancy_grid_map == 0] = [0, 0, 0] # Boş / Siyah (veya Gri)
        map_display[self.occupancy_grid_map == 255] = [255, 255, 255] # Dolu / Beyaz

        # Robotun anlık pozisyonunu haritada işaretle (küçük bir kırmızı daire)
        robot_map_x, robot_map_y = self._world_to_map(x, y)
        if 0 <= robot_map_x < self.map_pixel_size and 0 <= robot_map_y < self.map_pixel_size:
            cv2.circle(map_display, (robot_map_x, robot_map_y), 3, (0, 0, 255), -1) # Kırmızı nokta

        scale_factor = 2 
        map_display_resized = cv2.resize(map_display, 
                                         (self.map_pixel_size * scale_factor, self.map_pixel_size * scale_factor), 
                                         interpolation=cv2.INTER_NEAREST)
        
        cv2.imshow("Occupancy Grid Map", map_display_resized)
        
        return vis_frame

    def save_map_and_trajectory(self, map_filename="point_cloud_map.npy", traj_filename="trajectory.npy",
                                occupancy_map_filename="occupancy_grid_map.png"):
        if self.point_cloud_map:
            np.save(map_filename, np.array(self.point_cloud_map))
            print(f"Harita '{map_filename}' olarak kaydedildi. Toplam {len(self.point_cloud_map)} nokta.")
        else:
            print("Uyarı: 3D nokta bulutu haritası boş, kaydedilmedi.")

        if self.trajectory:
            np.save(traj_filename, np.array(self.trajectory))
            print(f"Yörünge '{traj_filename}' olarak kaydedildi. Toplam {len(self.trajectory)} pozisyon.")
        else:
            print("Uyarı: Yörünge boş, kaydedilmedi.")

        if self.occupancy_grid_map is not None:
            cv2.imwrite(occupancy_map_filename, self.occupancy_grid_map)
            print(f"İşgal ızgara haritası '{occupancy_map_filename}' olarak kaydedildi.")
        else:
            print("Uyarı: İşgal ızgara haritası boş, kaydedilmedi.")

    def plot_map_and_trajectory(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if self.point_cloud_map:
            map_points = np.array(self.point_cloud_map)
            ax.scatter(map_points[:, 0], map_points[:, 1], map_points[:, 2], s=0.1, c='blue', alpha=0.5, label='Nokta Bulutu Haritası')
        else:
            print("Uyarı: Görselleştirilecek 3D harita noktası yok.")

        if self.trajectory:
            traj_points = np.array(self.trajectory)
            ax.plot(traj_points[:, 0], traj_points[:, 1], traj_points[:, 2], 
                    color='red', linewidth=2, label='Kamera Yörüngesi')
            if len(traj_points) > 0:
                ax.scatter(traj_points[0, 0], traj_points[0, 1], traj_points[0, 2], s=50, c='green', marker='o', label='Başlangıç')
        else:
            print("Uyarı: Görselleştirilecek yörünge yok.")

        ax.set_xlabel('X Ekseni')
        ax.set_ylabel('Y Ekseni')
        ax.set_zlabel('Z Ekseni')
        ax.set_title('3D Nokta Bulutu Haritası ve Kamera Yörüngesi')
        ax.legend()
        plt.grid(True)
        plt.show()

# --- Kullanım Örneği ---
if __name__ == "__main__":
    try:
        slam = SIFT_SLAM("floor_segment_4temmuz.pt")
        print("YOLO modeli ve SIFT/BFMatcher başarıyla başlatıldı.")
    except Exception as e:
        print(f"HATA: SIFT_SLAM başlatılırken bir hata oluştu: {e}")
        print("Lütfen 'floor_segment_4temmuz.pt' dosyasının doğru yolda olduğundan ve ultralytics kütüphanesinin kurulu olduğundan emin olun.")
        exit()

    video_path = "depo_kisa.mp4"
    print(f"Video dosyası yolu: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"HATA: Video dosyası '{video_path}' açılamadı. Lütfen dosya yolunu, dosyanın varlığını ve okuma izinlerini kontrol edin.")
        print("Alternatif olarak, video kodeklerinde veya formatında bir sorun olabilir.")
        exit()

    print("Video başarıyla açıldı.")
    print(f"Video genişliği: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, Yüksekliği: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}, FPS: {cap.get(cv2.CAP_PROP_FPS)}")

    frame_count = 0
    print("Video işleniyor...")
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Video sonu, dosya okuma hatası veya daha fazla kare okunamadı.")
            break
        
        # Kare boyutunu küçültme (Performans için önerilir!)
        frame = cv2.resize(frame, (640, 480)) # Deneme için, farklı boyutlar deneyebilirsin
        # print(f"DEBUG: Kare {frame_count} okundu. Yeniden boyutlandırılmış Boyut: {frame.shape}")

        floor_mask, pose = slam.process_frame(frame)
        
        vis = slam.visualize(frame, floor_mask)
        cv2.imshow("SIFT-SLAM with Floor Segmentation (Original Frame)", vis)
        
        frame_count += 1
        if frame_count % 10 == 0: # Her 10 karede bir daha sık loglama
            print(f"İşlenen kare sayısı: {frame_count}. 3D Harita boyutu: {len(slam.point_cloud_map)} nokta. 2D Harita işaretli hücreler: {np.sum(slam.occupancy_grid_map == 255)}.")

        if cv2.waitKey(30) & 0xFF == ord('q'): 
            print("Kullanıcı çıkışı istendi.")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("Video işleme tamamlandı. Harita ve yörünge kaydediliyor...")
    slam.save_map_and_trajectory()
    print("Harita ve yörünge kaydedildi.")

    # 3D görselleştirme (isteğe bağlı)
    # slam.plot_map_and_trajectory() 
    print("Program tamamlandı.")