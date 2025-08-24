import cv2
import mediapipe as mp

# buat model deteksi wajah.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# # buat model deteksi tangan
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#   static_image_mode=False,
#   max_num_hands=2,
#   min_detection_confidence=0.5,
#   min_tracking_confidence=0.5
# )

capture = cv2.VideoCapture(0) # set default camera 0

# # gambar landmark tangan
# mp_drawing = mp.solutions.drawing_utils

# # ID landmark ujung jari
# finger_tips = [4, 8, 12, 16, 20]  # [jempol, telunjuk, tengah, manis, kelingking]


# set ukuran frame manual
width = capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
height = capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 850)

# get ukuran frame otomatis
# width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# kondisi
while True:

  ret, frame = capture.read()

  if not ret:
    break

  color = frame
  # color = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # # Process the frame to detect hands
  # results = hands.process(color)

  # # Check if hands are detected
  # if results.multi_hand_landmarks:
  #   for hand_landmarks in results.multi_hand_landmarks:
  #       # Draw landmarks on the frame
  #       mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

  #       # ambil semua titik
  #       landmarks = hand_landmarks.landmark
  #       h, w, _ = frame.shape
  #       fingers = []

  #       # cek jempol (pakai x-axis karena horizontal)
  #       if landmarks[finger_tips[0]].x < landmarks[finger_tips[0] - 1].x:
  #           fingers.append(1)  # terbuka
  #       else:
  #           fingers.append(0)  # tertutup

  #       # cek telunjuk sampai kelingking
  #       for tip_id in finger_tips[1:]:
  #           if landmarks[tip_id].y < landmarks[tip_id - 2].y:  
  #               fingers.append(1)
  #           else:
  #               fingers.append(0)

  #       total_fingers = fingers.count(1)

  #       cv2.putText(frame, f"Angka: {total_fingers}", (50, 100),
  #                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

  # cari wajah di gambar.
  faces = face_cascade.detectMultiScale(
    color,
    scaleFactor = 1.1,
    minNeighbors = 5,
    minSize = (30,30)
  )

  for (x,y,w,h) in faces:
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)  # bikin kotak merah di sekitar wajah (0,0,255) = merah.

    # ambil area wajah untuk deteksi senyum
    roi = color[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]

    smiles = smile_cascade.detectMultiScale(
      roi,
      scaleFactor = 1.7,
      minNeighbors = 22,
      minSize=(20,20)
    )

    ekspresi = ""
    warna = (255, 255, 255)

    if len(smiles) > 0:
      ekspresi = "Senyum"
      warna = (0, 255, 0)

      # cv2.putText(frame, "Senyum :", (x, y-10),
      #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    else:
      # coba deteksi mata untuk simulasi "marah"
      eyes = eye_cascade.detectMultiScale(roi, 1.1, 10, minSize=(15, 15))
            
      if len(eyes) >= 2:  
         # kalau mata jelas terlihat → anggap "Tidak Senyum"
        ekspresi = "Tidak Senyum"
        warna = (0, 255, 255)
      else:
        # kalau mata tidak jelas → anggap "Marah"
        ekspresi = "Marah"
        warna = (0, 0, 255)

    cv2.putText(frame, ekspresi,(x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, warna, 2)

  # jumlah wajah di frame
  cv2.putText(frame, f"Jumlah wajah: {len(faces)}", (10,30), 
              cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
  
  # cv2.putText(frame, f"Jumlah senyum: {len(smiles)}", (40,80),
  #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

  cv2.imshow('face recognition', frame)
  # cv2.imshow('frame', color)

  if cv2.waitKey(1) & 0xFF == 27:
    break

capture.release()
cv2.destroyAllWindows()