import cv2
import dlib
import numpy as np
from model import Net
import torch
from imutils import face_utils
from collections import deque
import time
import pygame
import pyttsx3

IMG_SIZE = (32, 26)
PATH = 'C:/weights/trained.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:/shape_predictor_68_face_landmarks.dat')

model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()

n_count = 0
blink_window = deque(maxlen=5)
start_time = None
accumulated_time = 0  # 눈을 감은 누적 시간을 저장하는 변수
drowsy_threshold = 0.15  # 졸음 판단을 위한 임계값


def tts(message):
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()


def play_alarm():
    alarm_path = 'C:/short_alarm.mp3'
    pygame.mixer.init()
    pygame.mixer.music.load(alarm_path)
    pygame.mixer.music.play()


def crop_eye(img, eye_points):
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 1.2
    h = w * IMG_SIZE[1] / IMG_SIZE[0]

    margin_x, margin_y = w / 2, h / 2

    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

    eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

    return eye_img, eye_rect


def predict(pred):
    pred = pred.transpose(1, 3).transpose(2, 3)

    outputs = model(pred)

    pred_tag = torch.round(torch.sigmoid(outputs))

    return pred_tag


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FPS, 0)


while cap.isOpened():
    ret, img_ori = cap.read()

    if not ret:
        break

    img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)

    img = img_ori.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        shapes = predictor(gray, face)
        shapes = face_utils.shape_to_np(shapes)

        eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
        eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

        eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
        eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
        eye_img_r = cv2.flip(eye_img_r, flipCode=1)

        cv2.imshow('l', eye_img_l)
        cv2.imshow('r', eye_img_r)

        eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)
        eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)

        eye_input_l = torch.from_numpy(eye_input_l)
        eye_input_r = torch.from_numpy(eye_input_r)

        pred_l = predict(eye_input_l)
        pred_r = predict(eye_input_r)

        if pred_l.item() == 0.0 and pred_r.item() == 0.0:
            n_count += 1
            if start_time is None:
                start_time = time.time()

            # 추가된 부분: 눈을 감은 누적 시간을 계산
            accumulated_time = time.time() - start_time

            # 추가된 부분: 누적 시간이 5초 이상이면서 눈을 감은 비율이 0.15 이상인 경우 졸음 상태로 판별
            if accumulated_time >= 5 and (n_count / accumulated_time) >= drowsy_threshold:
                cv2.putText(img, "Drowsy", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                play_alarm()

                tts("졸음 운전을 하고 있다 감지하여 간단한 수학 문제, 상식 문제, 역사 문제를 내드리겠습니다.")
        else:
            n_count = 0
            start_time = None
            accumulated_time = 0  # 눈을 뜬 경우 누적 시간 초기화

        blink_window.append(1 if n_count > 0 else 0)

        # 시각화
        state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
        state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'

        state_l = state_l % pred_l
        state_r = state_r % pred_r

        cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255, 255, 255), thickness=2)
        cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255, 255, 255), thickness=2)

        cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('result', img)

    # 프레임 값 출력#########################################
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps :", fps)
    # 프레임 값 출력#########################################

    cv2.waitKey(1)

    if cv2.waitKey(1) == ord('q'):
        break