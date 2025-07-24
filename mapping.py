# import cv2
# import numpy as np
# from ultralytics import YOLO

# class SIFT_SLAM:
#     def __init__(self, seg_model_path):
#         self.seg_model = YOLO(seg_model_path)
#         self.sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.04, edgeThreshold=10)
#         self.bf = cv2.BFMatcher()
#         self.last_frame = None
#         self.last_kp = None
#         self.last_des = None
#         self.trajectory = []
#         self.current_pose = np.eye(4)
        
#         # Harita için 3D nokta bulutu
#         self.point_cloud_map = [] 
        
#         # Kamera parametreleri (kendi kalibrasyon değerlerinizle değiştirin)
#         self.focal = 800
#         self.pp = (320, 240)
#         self.K = np.array([[self.focal, 0, self.pp[0]],
#                            [0, self.focal, self.pp[1]],
#                            [0, 0, 1]])

#     def process_frame(self, frame):
#         # 1. Zemin segmentasyonu
#         seg_results = self.seg_model(frame)
#         floor_mask = self._get_combined_mask(seg_results, frame.shape)
        
#         # 2. SIFT ile poz tahmini
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         kp, des = self.sift.detectAndCompute(gray, None)
        
#         if self.last_frame is not None and len(kp) > 20 and len(self.last_kp) > 20:
#             # 3. Eşleştirme yap
#             matches = self.bf.knnMatch(self.last_des, des, k=2)
            
#             # İyi eşleşmeleri filtrele
#             good = []
#             for m,n in matches:
#                 if m.distance < 0.75*n.distance:
#                     good.append(m)
            
#             if len(good) > 10:
#                 # 4. Poz tahmini ve 3D nokta kurtarma yap
#                 R, t, new_points_3d = self._estimate_pose(good, kp)
#                 if R is not None:
#                     self._update_trajectory(R, t)
#                     # Haritayı yeni 3D noktalarla güncelle
#                     if new_points_3d is not None:
#                         # Bu noktaları global koordinat sistemine dönüştür
#                         global_new_points_3d = (self.current_pose[:3, :3] @ new_points_3d.T).T + self.current_pose[:3, 3]
#                         self.point_cloud_map.extend(global_new_points_3d.tolist())
        
#         # Güncel frame'i bir sonraki adım için sakla
#         self.last_frame = gray
#         self.last_kp = kp
#         self.last_des = des
        
#         return floor_mask, self.current_pose

#     def _get_combined_mask(self, results, frame_shape):
#         combined_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
#         if results[0].masks is not None:
#             for mask in results[0].masks.data:
#                 mask_np = mask.cpu().numpy()
#                 mask_resized = cv2.resize(mask_np.astype('uint8'), 
#                                          (frame_shape[1], frame_shape[0]))
#                 combined_mask = cv2.bitwise_or(combined_mask, mask_resized)
#         return combined_mask

#     def _estimate_pose(self, matches, kp):
#         src_pts = np.float32([self.last_kp[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
#         dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        
#         # Esansiyel matris hesapla
#         E, mask = cv2.findEssentialMat(dst_pts, src_pts, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
#         if E is None:
#             return None, None, None
        
#         # Pozu ve 3D noktaları kurtar
#         # Not: recoverPose'un döndürdüğü 3D noktalar kamera 1'in koordinat sistemindedir.
#         _, R, t, triangulated_points = cv2.recoverPose(E, dst_pts, src_pts, self.K, mask=mask)
        
#         # triangulated_points dördüncü boyutu (homojen koordinat) ile gelir, normalize edelim
#         triangulated_points_3d = triangulated_points[:, :3] / triangulated_points[:, 3:]
        
#         return R, t, triangulated_points_3d

#     def _update_trajectory(self, R, t):
#         # Ölçek faktörü (gerçek dünya birimleri için kalibre edilmeli)
#         # Bu ölçek faktörü 3D noktaların derinliğini de etkiler
#         scale = 0.03 
        
#         # Dönüşüm matrisini oluştur
#         T = np.eye(4)
#         T[:3, :3] = R
#         T[:3, 3] = t.ravel() * scale # t'yi scale ile çarp
        
#         # Global pozisyonu güncelle
#         # current_pose = current_pose_önceki @ T_önceki_kare_göre_şimdi
#         # burada T, son_kareden_şuanki_kareye geçişi temsil ediyor
#         self.current_pose = self.current_pose @ np.linalg.inv(T)
#         self.trajectory.append(self.current_pose[:3, 3].copy())

#     def visualize(self, frame, floor_mask):
#         # Segmentasyon sonuçlarını görselleştir
#         vis = cv2.addWeighted(frame, 0.7, 
#                               cv2.cvtColor(floor_mask*255, cv2.COLOR_GRAY2BGR), 0.3, 0)
        
#         # Pozisyon bilgisini ekle
#         x, y, z = self.current_pose[:3, 3]
#         cv2.putText(vis, f"Position: X={x:.2f}, Y={y:.2f}, Z={z:.2f}", 
#                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
#         return vis

#     def save_map(self, filename="point_cloud_map.npy"):
#         """
#         Oluşturulan 3D nokta bulutu haritasını kaydeder.
#         """
#         np.save(filename, np.array(self.point_cloud_map))
#         print(f"Harita '{filename}' olarak kaydedildi.")

# # Kullanım örneği
# if __name__ == "__main__":
#     slam = SIFT_SLAM("floor_segment_4temmuz.pt")
#     cap = cv2.VideoCapture("depo_kisa.mp4")
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Frame'i işle
#         floor_mask, pose = slam.process_frame(frame)
        
#         # Görselleştir
#         vis = slam.visualize(frame, floor_mask)
#         cv2.imshow("SIFT-SLAM with Floor Segmentation", vis)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()
    
#     # Trajectory ve haritayı kaydet
#     np.save("trajectory.npy", np.array(slam.trajectory))
#     slam.save_map()

#     # Haritayı görselleştirmek için (isteğe bağlı, matplotlib gerektirir)
#     try:
#         import matplotlib.pyplot as plt
#         from mpl_toolkits.mplot3d import Axes3D

#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
        
#         # Nokta bulutu
#         map_points = np.array(slam.point_cloud_map)
#         if len(map_points) > 0:
#             ax.scatter(map_points[:, 0], map_points[:, 1], map_points[:, 2], s=1, c='blue')
        
#         # Trajectory
#         traj_points = np.array(slam.trajectory)
#         if len(traj_points) > 0:
#             ax.plot(traj_points[:, 0], traj_points[:, 1], traj_points[:, 2], color='red', linewidth=2)

#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')
#         ax.set_title('3D Point Cloud Map and Trajectory')
#         plt.show()

#     except ImportError:
#         print("Matplotlib 3D görselleştirme için yüklü değil. 'pip install matplotlib' ile yükleyebilirsiniz.")

import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SIFT_SLAM:
    def __init__(self, seg_model_path):
        """
        SIFT-tabanlı SLAM sistemini zemin segmentasyonu ile başlatır.

        Args:
            seg_model_path (str): YOLO segmentasyon modelinin yolu (örn: "floor_segment_4temmuz.pt").
        """
        self.seg_model = YOLO(seg_model_path)
        # SIFT özellik dedektörü ve tanımlayıcısı oluşturma
        # nfeatures: Bulunacak maksimum özellik sayısı
        # contrastThreshold: Özelliklerin ne kadar kontrastlı olması gerektiği
        # edgeThreshold: Kenarlarda özellik algılamayı azaltma
        self.sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.04, edgeThreshold=10)
        # Brute-Force Matcher (özellik eşleştirici) oluşturma
        self.bf = cv2.BFMatcher()

        # Önceki kareye ait veriler
        self.last_frame = None
        self.last_kp = None  # Anahtar noktalar (keypoints)
        self.last_des = None # Tanımlayıcılar (descriptors)

        # SLAM çıktısı için yörünge ve mevcut pozisyon
        self.trajectory = []
        self.current_pose = np.eye(4) # 4x4 Kimlik matrisi, başlangıç pozisyonu

        # 3D nokta bulutu haritası
        self.point_cloud_map = [] 
        
        # Kamera iç parametreleri (Kendi kalibrasyon değerlerinizle değiştirin!)
        # Focal: Odak uzaklığı (piksel cinsinden)
        # pp: Ana nokta (optik merkezin görüntüdeki koordinatları)
        self.focal = 800
        self.pp = (320, 240) # Genellikle görüntü genişliği/2, görüntü yüksekliği/2
        self.K = np.array([[self.focal, 0, self.pp[0]],
                           [0, self.focal, self.pp[1]],
                           [0, 0, 1]], dtype=np.float32) # Kamera matrisi

    def process_frame(self, frame):
        """
        Gelen bir video karesini işler, zemin segmentasyonu yapar,
        poz tahmini gerçekleştirir ve haritayı günceller.

        Args:
            frame (np.array): İşlenecek renkli görüntü karesi (BGR formatında).

        Returns:
            tuple: (floor_mask, current_pose)
                   floor_mask (np.array): Zemin segmentasyon maskesi (Gri tonlamalı).
                   current_pose (np.array): Robotun anlık 4x4 poz matrisi.
        """
        # 1. Zemin segmentasyonu (YOLO kullanılarak)
        seg_results = self.seg_model(frame, verbose=False) # verbose=False çıktıları azaltır
        floor_mask = self._get_combined_mask(seg_results, frame.shape)
        
        # 2. SIFT ile mevcut kare için anahtar nokta ve tanımlayıcı tespiti
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)
        
        # Sadece yeterli özellik varsa ve bir önceki kare varsa işleme devam et
        if self.last_frame is not None and kp is not None and des is not None and \
           self.last_kp is not None and self.last_des is not None and \
           len(kp) > 20 and len(self.last_kp) > 20: # Yeterli anahtar nokta kontrolü
            
            # 3. Önceki ve mevcut kareler arasındaki özellikleri eşleştirme
            matches = self.bf.knnMatch(self.last_des, des, k=2)
            
            # İyi eşleşmeleri filtrele (Lowe'un Oran Testi)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance: # Eşik değeri ayarlanabilir
                    good_matches.append(m)
            
            if len(good_matches) > 10: # Poz tahmini için yeterli eşleşme kontrolü
                # 4. Poz tahmini ve 3D nokta kurtarma
                R, t, new_points_3d = self._estimate_pose(good_matches, kp)
                
                if R is not None and t is not None:
                    self._update_trajectory(R, t)
                    
                    # Eğer 3D noktalar başarıyla triangüle edildiyse haritayı güncelle
                    if new_points_3d is not None and new_points_3d.shape[0] > 0:
                        # Triangüle edilen 3D noktaları global koordinat sistemine dönüştür
                        # new_points_3d kamera 1'in koordinat sistemindedir.
                        # self.current_pose global pozisyondadır.
                        # Dönüşüm: X_global = R_global_camera @ X_camera + t_global_camera
                        # Burada R_global_camera = self.current_pose[:3, :3]
                        # t_global_camera = self.current_pose[:3, 3]
                        
                        # Transpoz işlemi ve matris çarpımı için boyutların uyumlu olduğundan emin olun
                        # new_points_3d: (N, 3) -> transpose -> (3, N)
                        # self.current_pose[:3, :3]: (3, 3)
                        # Sonuç (3, N)
                        # Tekrar transpose -> (N, 3)
                        
                        # Poz matrisinin tersini kullanarak transformasyon yapılmalı
                        # current_pose = current_pose_önceki @ T_önceki_kare_göre_şimdi
                        # Bizim new_points_3d'miz R_son_kare_göre_şimdi, t_son_kare_göre_şimdi
                        # Eğer kamera koordinat sistemindeki 3D noktalarını,
                        # global koordinat sistemine taşımak istiyorsak
                        # global_point = R_global_kamera * camera_point + t_global_kamera
                        # Buradaki R ve t, current_pose'daki R ve t'dir.
                        
                        # Doğru dönüşüm: Dünya koordinatından kamera koordinatına
                        # P_cam = K[R|t]P_world
                        # Bizim triangulated_points_3d'miz kameranın kendi koordinat sisteminde.
                        # Bunu dünya koordinatına taşımak için current_pose'u kullanacağız.
                        # current_pose^-1 * [X_cam; 1]
                        
                        # Burada, _estimate_pose'dan gelen new_points_3d, önceki kameranın lokal sistemindedir.
                        # Bunu global harita sistemine dönüştürmek için current_pose kullanmak doğru değildir.
                        # Bunun yerine, eğer triangülasyon dünya koordinat sisteminde yapılsaydı,
                        # doğrudan ekleyebilirdik. recoverPose, iki kamera pozisyonu arasındaki
                        # bağıl dönüşümü (R, t) ve bu dönüşüme göre triangüle edilmiş noktaları verir.
                        # Bu noktalar genelde ilk kameranın koordinat sistemindedir.
                        
                        # Basit bir yaklaşımla, triangüle edilen noktaları mevcut kameranın global pozisyonuna göre
                        # dünya koordinat sistemine taşıyalım. Ancak bu, harita tutarlılığı için ideal değildir.
                        # Daha gelişmiş SLAM sistemleri burada Bundle Adjustment kullanır.
                        
                        # Geçici çözüm: Triangüle edilen noktaları son kameranın (current_pose)
                        # dünya koordinat sistemine taşıyalım.
                        # X_world = R_world_current_cam * X_current_cam + T_world_current_cam
                        
                        # current_pose, dünya koordinat sisteminin kameraya göre dönüşümüdür (T_WC).
                        # Bizim istediğimiz T_CW'dir. Yani T_WC'nin tersi.
                        # current_pose = [R_WC | t_WC]
                        # R_WC = self.current_pose[:3, :3]
                        # t_WC = self.current_pose[:3, 3]
                        
                        # Dönüşüm (current_pose, dünyanın kameraya göre dönüşümü olarak kabul ediliyor):
                        # P_world = R_WC * P_cam + t_WC
                        # Ancak self.current_pose'u T_CW olarak güncelliyoruz.
                        # Yani P_cam = T_CW * P_world
                        # P_world = T_WC * P_cam
                        # Yani P_world = current_pose * P_cam_homogen
                        
                        # Eğer current_pose robotun dünya sistemindeki pozisyonu ise (T_WR):
                        # World -> Camera: T_CW = T_WC_inverse = (T_RW)^-1
                        # Camera -> World: T_WC = T_RW
                        
                        # Mevcut kodda self.current_pose, dünyanın kameraya göre dönüşümüdür (T_WC).
                        # Dolayısıyla, kameranın koordinat sistemindeki noktaları dünya koordinatlarına çevirmek için:
                        R_wc = self.current_pose[:3, :3]
                        t_wc = self.current_pose[:3, 3].reshape(3, 1) # reshape to (3,1) for broadcasting
                        
                        global_new_points_3d = (R_wc @ new_points_3d.T).T + t_wc.T 
                        
                        self.point_cloud_map.extend(global_new_points_3d.tolist())
                    else:
                        # print("Uyarı: 'cv2.recoverPose' ile 3D nokta triangüle edilemedi veya boş.")
                        pass # Sık sık olabilir, her seferinde uyarı vermek yerine sessiz kalabiliriz
                else:
                    # print("Uyarı: Poz tahmini başarısız oldu (R veya t 'None').")
                    pass
            else:
                # print(f"Uyarı: Yeterli iyi eşleşme yok ({len(good_matches)}). Poz tahmini atlandı.")
                pass
        else:
            # print(f"Uyarı: Yeterli anahtar nokta yok veya ilk kare. Anahtar nokta sayısı: {len(kp) if kp is not None else 0}, Son anahtar nokta sayısı: {len(self.last_kp) if self.last_kp is not None else 0}")
            pass

        # Güncel kareyi bir sonraki adım için sakla
        # SIFT'in yeniden hesaplama yapmaması için gri tonlamalı frame saklanır
        self.last_frame = gray
        self.last_kp = kp
        self.last_des = des
        
        return floor_mask, self.current_pose

    def _get_combined_mask(self, results, frame_shape):
        """
        YOLO segmentasyon sonuçlarından birleşik bir maske oluşturur.

        Args:
            results (ultralytics.engine.results.Results): YOLO modelinin çıktıları.
            frame_shape (tuple): Orijinal çerçevenin şekli (height, width, channels).

        Returns:
            np.array: Birleşik ikili segmentasyon maskesi (np.uint8).
        """
        combined_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        if results[0].masks is not None:
            for mask in results[0].masks.data:
                mask_np = mask.cpu().numpy()
                # Maskeyi orijinal çerçeve boyutuna yeniden boyutlandır
                mask_resized = cv2.resize(mask_np.astype('uint8'), 
                                         (frame_shape[1], frame_shape[0]))
                combined_mask = cv2.bitwise_or(combined_mask, mask_resized)
        return combined_mask

    def _estimate_pose(self, matches, current_kp):
            """
            Eşleşen anahtar noktaları kullanarak Esansiyel Matrisi, kamera pozunu (R, t) tahmin eder
            ve bu noktalardan 3D konumlarını triangüle eder.

            Args:
                matches (list): `cv2.BFMatcher.knnMatch`'ten gelen iyi eşleşmeler listesi.
                current_kp (list): Mevcut kareye ait anahtar noktalar.

            Returns:
                tuple: (R, t, new_points_3d)
                    R (np.array): 3x3 Dönme matrisi.
                    t (np.array): 3x1 Öteleme vektörü.
                    new_points_3d (np.array): Triangüle edilmiş 3D noktalar (Nx3).
                    Eğer tahmin başarısız olursa (None, None, None) döner.
            """
            # Eşleşen noktaları ayır: önceki kare (sorgu noktaları) ve mevcut kare (eğitim noktaları)
            # cv2.findEssentialMat, points1 (image1) ve points2 (image2) bekler.
            # Bizim senaryomuzda: image1 = current_frame (dst_pts), image2 = last_frame (src_pts)
            # Bu yüzden dst_pts'yi ilk parametre, src_pts'yi ikinci parametre olarak veriyoruz.
            
            src_pts_np = np.float32([self.last_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts_np = np.float32([current_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # print(f"DEBUG: src_pts_np shape before findEssentialMat: {src_pts_np.shape}")
            # print(f"DEBUG: dst_pts_np shape before findEssentialMat: {dst_pts_np.shape}")

            # Esansiyel matris hesapla (RANSAC kullanarak aykırı değerleri elemek için)
            # cv2.findEssentialMat(img1_points, img2_points, ...)
            E, mask = cv2.findEssentialMat(dst_pts_np, src_pts_np, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            
            if E is None or E.shape == (0, 0): # Essential matris bulunamazsa veya boşsa
                # print("Warning: Essential Matrix estimation failed or returned empty E.")
                return None, None, None
            
            # print(f"DEBUG: Essential Matrix E shape: {E.shape}")
            # print(f"DEBUG: Mask shape from findEssentialMat: {mask.shape}, dtype: {mask.dtype}")
            # print(f"DEBUG: Number of inliers (mask sum): {np.sum(mask)}")

            # Sağlam poz tahmini için en az 8 iç nokta gerekir
            if np.sum(mask) < 8: 
                # print(f"Warning: Not enough inliers ({np.sum(mask)}) for robust pose recovery. Skipping triangulation.")
                return None, None, None 

            # Pozu kurtar (Sadece R ve t için recoverPose kullanıyoruz)
            # Bu R ve t, mevcut karenin (dst_pts) önceki kareye (src_pts) göre bağıl dönüşümüdür.
            # recoverPose(E, img1_points, img2_points, ...)
            _, R, t, _ = cv2.recoverPose(E, dst_pts_np, src_pts_np, self.K, mask=mask)
            
            if R is None or t is None: # R veya t başarıyla kurtarılamazsa
                # print("Warning: Pose recovery failed (R or t is None).")
                return None, None, None

            # --- Manuel Triangülasyon: cv2.triangulatePoints kullanarak 3D nokta kurtarma ---
            # 1. Her iki kamera için projeksiyon matrislerini oluştur
            # Birinci kamera (önceki kare - last_frame) dünya koordinat sisteminin orijinindedir: P1 = K * [I | 0]
            P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1)))) 
            
            # İkinci kamera (mevcut kare - current_frame) R ve t ile tanımlanır: P2 = K * [R | t]
            # Burada R ve t, önceki kareye göre olan bağıl poz dönüşümüdür.
            P2 = self.K @ np.hstack((R, t)) 

            # 2. findEssentialMat'ten gelen maskeyi kullanarak sadece iç nokta (inlier) eşleşmelerini al
            # cv2.triangulatePoints, 2D noktaları 2xN formatında bekler.
            # reshape(-1, 2).T ile Nx1x2'den 2xN'e dönüştürüyoruz.
            src_pts_inliers = src_pts_np[mask.ravel() == 1].reshape(-1, 2).T
            dst_pts_inliers = dst_pts_np[mask.ravel() == 1].reshape(-1, 2).T

            if src_pts_inliers.shape[1] == 0: # Eğer hiç iç nokta kalmazsa
                # print("Warning: No inlier points for triangulation after mask application.")
                return R, t, None # Pozu döndür, ancak 3D nokta döndürme

            # 3. Triangülasyonu gerçekleştir
            # Sonuç: 4xN homojen koordinatlar (P1'in koordinat sisteminde)
            triangulated_homogeneous_points = cv2.triangulatePoints(P1, P2, src_pts_inliers, dst_pts_inliers)
            
            # Triangüle edilmiş noktalar boşsa
            if triangulated_homogeneous_points.shape[1] == 0:
                # print("Warning: cv2.triangulatePoints returned no points.")
                return R, t, None

            # 4. Homojen koordinatlardan Öklid (3D) koordinatlara dönüştür
            # Sonuç 4xN olduğu için, Nx4 yapmak için transpoz alıyoruz.
            triangulated_homogeneous_points = triangulated_homogeneous_points.T 

            # w değeri sıfıra yakınsa veya negatifse, bölme hatası veya anlamsız sonuçlar verir.
            # Sadece pozitif ve yeterince büyük w değerlerine sahip noktaları al.
            valid_indices = triangulated_homogeneous_points[:, 3] > 1e-6 
            triangulated_points_3d = triangulated_homogeneous_points[valid_indices, :3] / triangulated_homogeneous_points[valid_indices, 3:]

            # 5. Derinlik kontrolü ve gürültü filtrelemesi
            # Genellikle Z ekseni ileriye doğrudur. Negatif derinlikleri eleyelim (kameranın arkasındaki noktalar).
            # Ayrıca, çok uzak veya çok yakın noktaları da filtreleyebiliriz.
            # Örneğin, 0.1 metreden yakın veya 100 metreden uzak noktaları ele.
            valid_depth_indices = (triangulated_points_3d[:, 2] > 0.1) & (triangulated_points_3d[:, 2] < 100)
            triangulated_points_3d = triangulated_points_3d[valid_depth_indices]
            
            if triangulated_points_3d.shape[0] == 0:
                # print("Warning: No 3D points remaining after depth filtering.")
                return R, t, None

            # print(f"DEBUG: Number of successfully triangulated 3D points: {triangulated_points_3d.shape[0]}")
            return R, t, triangulated_points_3d

    def _update_trajectory(self, R, t):
        """
        Kamera pozunu günceller ve yörüngeye ekler.

        Args:
            R (np.array): Geçerli kareye göre dönme matrisi.
            t (np.array): Geçerli kareye göre öteleme vektörü.
        """
        # Ölçek faktörü (önemli! Gerçek dünya birimleri için kalibre edilmeli)
        # Bu değer, t (öteleme vektörü) üzerinde bir etkiye sahiptir ve
        # haritanın genel büyüklüğünü belirler.
        # Genellikle ilk karelerde veya bilinen bir referansla kalibre edilir.
        scale = 0.03 # Varsayımsal bir değer. Genellikle 1 birim ~ 1 metre olarak ayarlanmaya çalışılır.
        
        # Göreceli dönüşüm matrisini oluştur (önceki kareden mevcut kareye)
        # T_prev_curr = [R | t]
        #           = [R | t*scale]
        #           [0 0 0 | 1]
        T_prev_curr = np.eye(4)
        T_prev_curr[:3, :3] = R
        T_prev_curr[:3, 3] = t.ravel() * scale
        
        # Global pozisyonu güncelle
        # current_pose_global = current_pose_global_önceki @ T_prev_curr
        # Ancak recoverPose'dan gelen R, t, kamera 1'den kamera 2'ye dönüşümü veriyorsa (T_12).
        # current_pose, dünya koordinat sisteminden kameranın mevcut pozisyonuna dönüşümü temsil ediyorsa (T_WC).
        # O zaman yeni kamera pozisyonu (T_WC_new) = T_WC_old @ T_C1_C2
        # Burada T_C1_C2 = [R_12 | t_12]
        
        # Not: cv2.recoverPose, 2. kameranın (şimdiki kare) 1. kameraya (önceki kare) göre pozisyonunu verir.
        # Yani R ve t, T_last_current olarak düşünülebilir (current frame'in last frame'e göre pozisyonu).
        # self.current_pose = T_World_LastCamera
        # self.new_current_pose = T_World_LastCamera @ T_LastCamera_CurrentCamera
        # T_World_CurrentCamera = T_World_LastCamera @ T_LastCamera_CurrentCamera
        # Bu yüzden `np.linalg.inv(T)` yerine doğrudan `T_prev_curr` kullanmalıyız.
        
        # Eğer current_pose, dünyanın kameraya göre dönüşümüyse (T_WC),
        # yeni pozisyon: T_WC_new = T_WC_old @ T_C_C_new
        # where T_C_C_new is the transformation from current to new camera.
        # Since recoverPose gives the transformation from first camera (last frame) to second camera (current frame),
        # this is T_Last_Current.
        # So, T_World_Current = T_World_Last @ T_Last_Current
        self.current_pose = self.current_pose @ T_prev_curr 
        
        self.trajectory.append(self.current_pose[:3, 3].copy())

    def visualize(self, frame, floor_mask):
        """
        Giriş karesini segmentasyon maskesi ve pozisyon bilgisiyle görselleştirir.

        Args:
            frame (np.array): Orijinal renkli görüntü karesi.
            floor_mask (np.array): Zemin segmentasyon maskesi.

        Returns:
            np.array: Görselleştirilmiş görüntü.
        """
        # Segmentasyon sonuçlarını orijinal frame üzerine bindir
        # Zemin maskesini renkli bir görüntüye dönüştür ve opaklık ekle
        vis = cv2.addWeighted(frame, 0.7, 
                              cv2.cvtColor(floor_mask * 255, cv2.COLOR_GRAY2BGR), 0.3, 0)
        
        # Pozisyon bilgisini görüntüye ekle
        x, y, z = self.current_pose[:3, 3] # x, y, z koordinatları
        cv2.putText(vis, f"Position: X={x:.2f}, Y={y:.2f}, Z={z:.2f}", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis

    def save_map_and_trajectory(self, map_filename="point_cloud_map.npy", traj_filename="trajectory.npy"):
        """
        Oluşturulan 3D nokta bulutu haritasını ve yörüngeyi NumPy dosyaları olarak kaydeder.

        Args:
            map_filename (str): Harita dosyasının adı.
            traj_filename (str): Yörünge dosyasının adı.
        """
        if self.point_cloud_map:
            np.save(map_filename, np.array(self.point_cloud_map))
            print(f"Harita '{map_filename}' olarak kaydedildi. Toplam {len(self.point_cloud_map)} nokta.")
        else:
            print("Uyarı: Harita boş, kaydedilmedi.")

        if self.trajectory:
            np.save(traj_filename, np.array(self.trajectory))
            print(f"Yörünge '{traj_filename}' olarak kaydedildi. Toplam {len(self.trajectory)} pozisyon.")
        else:
            print("Uyarı: Yörünge boş, kaydedilmedi.")

    def plot_map_and_trajectory(self):
        """
        Oluşturulan 3D nokta bulutu haritasını ve yörüngeyi Matplotlib kullanarak görselleştirir.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Nokta bulutu haritasını çiz
        if self.point_cloud_map:
            map_points = np.array(self.point_cloud_map)
            # Yalnızca belirli bir aralıktaki noktaları çizerek görselleştirmeyi hızlandırabiliriz
            # Örneğin, çok sayıda nokta varsa:
            # step = max(1, len(map_points) // 100000) # 100,000 noktayı geçmez
            # ax.scatter(map_points[::step, 0], map_points[::step, 1], map_points[::step, 2], s=0.1, c='blue', alpha=0.5)
            ax.scatter(map_points[:, 0], map_points[:, 1], map_points[:, 2], s=0.1, c='blue', alpha=0.5, label='Nokta Bulutu Haritası')
        else:
            print("Uyarı: Görselleştirilecek harita noktası yok.")

        # Yörüngeyi çiz
        if self.trajectory:
            traj_points = np.array(self.trajectory)
            ax.plot(traj_points[:, 0], traj_points[:, 1], traj_points[:, 2], 
                    color='red', linewidth=2, label='Kamera Yörüngesi')
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
    # SLAM sınıfını başlat
    # Dikkat: "floor_segment_4temmuz.pt" model dosyasının mevcut olduğundan emin olun!
    try:
        slam = SIFT_SLAM("floor_segment_4temmuz.pt")
    except Exception as e:
        print(f"YOLO modelini yüklerken bir hata oluştu: {e}")
        print("Lütfen 'floor_segment_4temmuz.pt' dosyasının doğru yolda olduğundan ve ultralytics kütüphanesinin kurulu olduğundan emin olun.")
        exit() # Hata durumunda programı sonlandır

    # Video dosyasını aç
    video_path = "depo_kisa.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Hata: Video dosyası '{video_path}' açılamadı.")
        exit()

    frame_count = 0
    print("Video işleniyor...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video sonu veya okuma hatası.")
            break
        
        # Frame'i işle
        floor_mask, pose = slam.process_frame(frame)
        
        # Görselleştir
        vis = slam.visualize(frame, floor_mask)
        cv2.imshow("SIFT-SLAM with Floor Segmentation", vis)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"İşlenen kare sayısı: {frame_count}. Harita boyutu: {len(slam.point_cloud_map)} nokta.")

        # 'q' tuşuna basıldığında çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Kullanıcı çıkışı istendi.")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Harita ve yörüngeyi kaydet
    slam.save_map_and_trajectory()

    # Harita ve yörüngeyi görselleştir
    print("Harita ve yörünge görselleştiriliyor...")
    slam.plot_map_and_trajectory()
    print("Program tamamlandı.")