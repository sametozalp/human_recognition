import cv2
import torch
from tracker import *
import face_recognition

people = {
    "Elon": "images/elon.jpg", 
    "Samet": "images/sametozalp.jpg"
}

known_encodings = {}
for name, image_path in people.items():
    known_image = face_recognition.load_image_file(image_path)
    known_encoding = face_recognition.face_encodings(known_image)[0]
    known_encodings[name] = known_encoding

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
cap=cv2.VideoCapture(0)
tracker = Tracker()

while True:
    ret,frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame,(1920,1080))
    
    results = model(frame)
    
    list=[]
    
    for index,row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        b = str(row['name'])
        if 'person' in b:
             list.append([x1,y1,x2,y2])
    
    kisi_sayisi = tracker.update(list)
    
    for box_id in kisi_sayisi:
        x,y,w,h,id=box_id
        cv2.rectangle(frame, (x,y), (w,h), (0,255,0), 3)
    
    print("Kişi Sayisi: ", len(kisi_sayisi))  
    
    cv2.putText(frame, "Kisi Sayisi: " + str(len(kisi_sayisi)), (20,60), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)
    
    ####################################################################################################33
    
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_location, face_encoding in zip(face_locations, face_encodings):
        for name, known_encoding in known_encodings.items():
            matches = face_recognition.compare_faces([known_encoding], face_encoding)
            if True in matches:
                
                identified_name = name
                break
        else:
            identified_name = "Bilinmiyor"

        # Yüzü dikdörtgen içine al
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        cv2.putText(frame, identified_name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    cv2.imshow('DART',frame)
    if cv2.waitKey(1) & 0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()