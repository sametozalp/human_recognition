import cv2
import torch
import face_recognition
from tracker import Tracker

# YOLOv5 modelini yükle
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Kamerayı aç
cap = cv2.VideoCapture(0)

# Tracker'ı başlat
tracker = Tracker()

# Tanınacak kişinin fotoğrafını yükle ve kodla
saved_img = face_recognition.load_image_file('elon.jpg')
saved_img_encoding = face_recognition.face_encodings(saved_img)[0]

while True:
    # Kameradan görüntü al
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLOv5 ile nesne tespiti yap
    results = model(frame)
    
    # YOLOv5 sonuçlarını işle
    list = []
    for index, row in results.pandas().xyxy[0].iterrows():
        if row['name'] == 'person':
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            list.append([x1, y1, x2, y2])
    
    # Yüz tanıma yap
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    
    # Eşleşen yüzleri bul
    matched_faces = []
    for face_encoding in face_encodings:
        match = face_recognition.compare_faces([saved_img_encoding], face_encoding)
        matched_faces.append(match[0])
    
    # Tanınan yüzlerin koordinatlarını ve eşleşme durumunu birleştir
    for (top, right, bottom, left), match in zip(face_locations, matched_faces):
        if match:
            list.append([left, top, right, bottom])
    
    # Yüz ve kişi sayılarını takip et
    tracked_boxes = tracker.update(list)
    num_people = len(tracked_boxes)
    
    # Tanılan kişi sayısını çerçeveye yazdır
    cv2.putText(frame, f"Kişi Sayısı: {num_people}", (20, 60), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    
    # Kişilerin etrafına dikdörtgen çiz
    for box_id in tracked_boxes:
        x, y, w, h, index = box_id
        cv2.rectangle(frame, (x, y), (w, h), (255, 0, 255), 2)
    
    # Sonuçları göster
    cv2.imshow('DART', frame)
    
    # Çıkış için 'ESC' tuşuna basılmasını bekleyin
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Kamerayı serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
