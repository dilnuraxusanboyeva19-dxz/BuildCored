import cv2
import mediapipe as mp
import pygame
import time

pygame.mixer.init()
music_path = r"C:\Users\Dilnura\Downloads\Wonder Girls - Tell Me.mp3"
pygame.mixer.music.load(music_path)
pygame.mixer.music.play(-1)
pygame.mixer.music.pause()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(0) 
state = "UP"
last_change_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        continue 

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose_y = int(face_landmarks.landmark[1].y * h)
            
            if nose_y > h * 0.6:
                new_state = "DOWN"
            else:
                new_state = "UP"

            if new_state != state and (time.time() - last_change_time > 0.5):
                state = new_state
                last_change_time = time.time()
                if state == "DOWN":
                    pygame.mixer.music.unpause()
                else:
                    pygame.mixer.music.pause()

    cv2.putText(frame, f"State: {state}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("RockLook Demo", frame)

    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
