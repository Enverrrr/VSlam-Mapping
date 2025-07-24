# from ultralytics import YOLO

# # Load a model
# # model = YOLO("yolo11n-seg.pt")  # load an official model
# model = YOLO("floor_segment_4temmuz.pt")  # load a custom model

# # Predict with the model
# results = model("depo_kisa.mp4")  # predict on a video file

# for result in results:
#     xy = result.masks.xy  # mask in polygon format
#     xyn = result.masks.xyn  # normalized
#     masks = result.masks.data  # mask in matrix format (num_objects x H x W)


from ultralytics import YOLO
import cv2
import numpy as np

# Modeli yükle
model = YOLO("floor_segment_4temmuz.pt")  # Özel zemin segmentasyon modeliniz

# Video yakalayıcıyı başlat
video_path = "depo_kisa.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLO ile segmentasyon yap
    results = model(frame)
    
    # Orijinal frame'i kopyala
    display_frame = frame.copy()
    
    for result in results:
        # Maske varsa işle
        if result.masks is not None:
            # Tüm maskeleri birleştir
            combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            for mask in result.masks.data:
                # Maskeyi CPU'ya taşı ve numpy array'e çevir
                mask_np = mask.cpu().numpy()
                # Maskeyi orijinal boyuta yeniden boyutlandır
                mask_resized = cv2.resize(mask_np.astype('uint8'), 
                                        (frame.shape[1], frame.shape[0]))
                # Maskeleri birleştir
                combined_mask = cv2.bitwise_or(combined_mask, mask_resized)
            
            # Segmentasyon sonuçlarını görselleştir
            # 1. Maskeyi renklendir
            color_mask = np.zeros_like(frame)
            color_mask[combined_mask == 1] = [0, 255, 0]  # Yeşil renk
            
            # 2. Orijinal görüntüyle maskeyi birleştir
            display_frame = cv2.addWeighted(display_frame, 0.7, color_mask, 0.3, 0)
            
            # 3. Sınırları çiz
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display_frame, contours, -1, (0, 0, 255), 2)  # Kırmızı sınırlar
    
    # Sonuçları göster
    cv2.imshow('Floor Segmentation', display_frame)
    
    # Çıkış için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()