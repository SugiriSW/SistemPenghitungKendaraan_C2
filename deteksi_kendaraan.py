from ultralytics import YOLO
import cv2
import os
import numpy as np

# Path Model dan Video
model_path = r"D:\Kuliah\Semester_5\Comvis\TUBES\TUBES\vehicle-counting-7\runs\detect\train7\weights\best.pt"
video_source = r"D:\Kuliah\Semester_5\Comvis\TUBES\TUBES\video1.mp4"

# Validasi File
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model tidak ditemukan di: {model_path}")
if not os.path.exists(video_source):
    raise FileNotFoundError(f"Video tidak ditemukan di: {video_source}")

# Load Model YOLOv8
model = YOLO(model_path)

# Buka Video
cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    raise RuntimeError("Gagal membuka video.")

# Output Video
output_video_path = r"E:\Kampus\Semester 5\Computer Vision\TUBES\vehicle-counting-7\result.avi"
result = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 20,
                        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Penghitung kendaraan
incount1 = 0  # Mobil
incount2 = 0  # Motor
incount3 = 0  # Bus
incount4 = 0  # Truk
incount5 = 0  # Total

# Definisikan Bounding Box Jalan 
#road_box = [200, 100, 800, 300]  # Boudning Box Video5, Video6, video7
# road_box = [50, 100, 400, 350] #Bounding Box Video10
#road_box = [200, 300, 1000, 500]  # Bounding Box video4
road_box = [50, 50, 600, 400]  # Bounding Box video1, video2, video3, video8, video9, video11

# Variabel untuk menyimpan objek yang sudah terdeteksi beserta ID-nya
tracked_objects = {}

# Threshold Confidence
confidence_threshold = 0.3
frame_count = 0  # Variabel untuk melacak jumlah frame berturut-turut

# Threshold untuk jarak antara centroid objek
centroid_distance_threshold = 50  # Anda bisa sesuaikan sesuai kebutuhan

# Fungsi untuk memeriksa apakah kendaraan melewati bounding box jalan
def has_crossed_road(centroid, road_box):
    x1, y1, x2, y2 = road_box
    cx, cy = centroid
    if y1 < cy < y2 and x1 < cx < x2:
        return True
    return False

def draw_road_box(frame, road_box):
    x1, y1, x2, y2 = road_box
    # Gambar kotak merah dengan ketebalan 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Fungsi untuk menghitung IoU (Overlap antara 2 kotak)
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    xi1, yi1 = max(x1, x1b), max(y1, y1b)
    xi2, yi2 = min(x2, x2b), min(y2, y2b)
    intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2b - x1b) * (y2b - y1b)
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area

# Fungsi untuk mendapatkan centroid dari bounding box
def get_centroid(x1, y1, x2, y2):
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

# Fungsi untuk menghitung jarak antara dua titik centroid
def compute_centroid_distance(c1, c2):
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

# Fungsi untuk memeriksa apakah objek sudah terdeteksi sebelumnya
def is_duplicate_object(new_centroid, new_box):
    for obj_id, obj_data in tracked_objects.items():
        prev_centroid, prev_box = obj_data[:2]
        # Periksa IoU dan jarak centroid untuk menentukan apakah ini objek yang sama
        iou = compute_iou(new_box, prev_box)
        if iou > 0.3 and compute_centroid_distance(new_centroid, prev_centroid) < centroid_distance_threshold:
            return obj_id  # Objek ditemukan, return ID yang sama
    return None  # Jika objek baru, return None

try:
    # Loop Utama
    crossed_vehicles = set()  # Menyimpan ID kendaraan yang sudah lewat
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Gambar road_box (garis merah)
        draw_road_box(frame, road_box)

        # Proses hanya setiap 3 frame
        if frame_count % 3 == 0:
            # Inference dengan YOLOv8
            results = model(frame)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    cls_name = model.names[cls]

                    if conf < confidence_threshold:
                        continue

                    # Hitung centroid dari bounding box
                    centroid = get_centroid(x1, y1, x2, y2)

                    # Cek apakah objek berada di dalam road_box
                    if has_crossed_road(centroid, road_box):
                        # Cek apakah objek sudah terdeteksi sebelumnya
                        obj_id = is_duplicate_object(centroid, (x1, y1, x2, y2))

                        if obj_id is not None:
                            # Jika objek ditemukan duplikat, perbarui data objek yang ada
                            tracked_objects[obj_id] = (centroid, (x1, y1, x2, y2), frame_count)
                        else:
                            # Jika objek baru, beri ID unik dan simpan
                            new_id = len(tracked_objects) + 1  # ID unik baru
                            tracked_objects[new_id] = (centroid, (x1, y1, x2, y2), frame_count)  # Simpan objek baru

                            # Hitung kategori kendaraan
                            if cls_name == 'car' or cls_name == 'mobil':
                                incount1 += 1
                            elif cls_name == 'motorcycle' or cls_name == 'motor':
                                incount2 += 1
                            elif cls_name == 'bus':
                                incount3 += 1
                            elif cls_name == 'truck' or cls_name == 'truk':
                                incount4 += 1

                        # Kotak Deteksi
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Update frame count
        frame_count += 1

        # Hitung Total
        incount5 = incount1 + incount2 + incount3 + incount4

        # Tampilkan Hasil
        cv2.putText(frame, f'Mobil : {incount1}', (25, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 0), 2)
        cv2.putText(frame, f'Motor : {incount2}', (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        cv2.putText(frame, f'Bus : {incount3}', (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
        cv2.putText(frame, f'Truk : {incount4}', (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
        cv2.putText(frame, f'Total : {incount5}', (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

        result.write(frame)
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"Terjadi kesalahan: {e}")
finally:
    cap.release()
    result.release()
    cv2.destroyAllWindows()
