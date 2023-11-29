from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import pygame
import pyttsx3
import os
import openai
from dotenv import load_dotenv
import speech_recognition as sr
from gtts import gTTS

openai.api_key = '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'
load_dotenv()
model = 'gpt-3.5-turbo'

# Set up the speech recognition and text-to-speech engines
r = sr.Recognizer()
engine = pyttsx3.init("dummy")
voice = engine.getProperty('voices')[1]
engine.setProperty('voice', voice.id)


def listen_and_respond(audio_text):
    print(f"You said: {audio_text}")
    if not audio_text:
        return

    # ChatGPT의 응답에서 ? 나 = 뒤의 부분 제거
    response_text = audio_text.split('?')[0].strip()
    print(response_text)

    response = openai.Completion.create(
        engine="text-davinci-003",  # GPT-3.5-turbo 엔진 사용
        prompt=f"User: {response_text}\nAI:",
        max_tokens=500
    )

    full_response_text = response.choices[0].text.split('?')[0].split('=')[0].strip()
    print(full_response_text)

    # Convert text response to speech
    print("generating audio")
    myobj = gTTS(text=full_response_text, lang='ko', slow=False)
    save_directory = os.path.join(os.path.expanduser("~"), "Desktop")

    # Generate a unique filename based on the current timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")
    file_name = f"response_{timestamp}.mp3"
    file_path = os.path.join(save_directory, file_name)

    myobj.save(file_path)
    print("speaking")
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    # Speak the response
    print("speaking complete")


# 눈 종횡비 구하기
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# PERCLOS를 나타냄
# 프레임이 5초 동안 PERCLOS 값이 0.15 이상이면 알람을 울림
PERCLOS_THRESHOLD = 0.15
PERCLOS_INTERVAL = 3  # 3초 동안 PERCLOS 값을 측정
PERCLOS_FRAMES = []
ALARM_ON = False

# 파일이 로컬됐는지 확인
print("[INFO] 파일이 로컬됨...")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/jetson/cleancode/shape_predictor_68_face_landmarks.dat")

# 왼쪽과 오른쪽 눈의 인덱스를 잡음
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# 비디오가 시작되는지 확인
print("[INFO] 비디오 시작 중...")
vs = VideoStream(src=0).start()  # src를 0으로 변경
time.sleep(1.0)

# 초기화
pygame.mixer.init()
pygame.mixer.music.load('/home/jetson/cleancode/short_alarm.mp3')


def tts(message):
    global rects
    tts = gTTS(text=message, lang='ko')
    print("2")
    filename = 'voice.mp3'
    tts.save(filename)
    pygame.mixer.init()
    pygame.mixer.music.load('/home/jetson/voice.mp3')
    pygame.mixer.music.play()


# 비디오 스트림의 프레임 반복
while True:
    frame = vs.read()

    if frame is None:
        print("[INFO] 프레임을 읽지 못했습니다.")
        continue  # 다음 반복으로 넘어감

    frame = imutils.resize(frame, width=320, height=260)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # 양 쪽 눈의 눈 종횡비를 계산하는 좌표
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # 양쪽 눈의 눈 종횡비 평균을 합산
        ear = (leftEAR + rightEAR) / 2.0

        # 각 눈을 시각화
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(gray, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(gray, [rightEyeHull], -1, (0, 255, 0), 1)

        # 눈 감음 여부에 따라 텍스트 표시
        eye_state = "1" if ear > PERCLOS_THRESHOLD else "0"
        cv2.putText(gray, f"Eye State: {eye_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 현재 프레임의 EAR 값을 PERCLOS_FRAMES에 추가
        PERCLOS_FRAMES.append(ear)

        # PERCLOS_FRAMES를 체크하여 5초 동안의 PERCLOS 값을 계산
        if len(PERCLOS_FRAMES) >= PERCLOS_INTERVAL * 60:  # 30fps 가정
            perclos_value = sum(1 for e in PERCLOS_FRAMES if e < PERCLOS_THRESHOLD) / len(PERCLOS_FRAMES)
            print("PERCLOS:", perclos_value)

            # PERCLOS 값이 임계값보다 크면 알람 울림
            if perclos_value >= PERCLOS_THRESHOLD:
                if not ALARM_ON:
                    ALARM_ON = True
                    pygame.mixer.music.play()
                    tts("졸음 운전을 하고 있다 감지하여 간단한 수학 문제, 상식 문제, 역사 문제를 내드리겠습니다.")

                    with sr.Microphone() as source:
                        print("Listening...")
                        while True:
                            audio = r.listen(source)
                            try:
                                audio_text = r.recognize_google(audio, language="ko-KR-Neural2-A")
                                listen_and_respond(audio_text)

                                # 추가: 사용자가 '종료'라고 말하면 루프를 종료하고 프로그램을 종료
                                if '종료' in audio_text:
                                    break

                            except sr.UnknownValueError:
                                pass

                        # 추가: 사용자가 '종료'라고 말한 경우 루프를 종료하고 프로그램을 종료
                    if '종료' in audio_text:
                        break

            PERCLOS_FRAMES = []  # PERCLOS_FRAMES 초기화

    cv2.imshow("dahaebozo", gray)
    key = cv2.waitKey(1) & 0xFF

    # q를 눌러 종료
    if key == ord("q"):
        break

cv2.destroyAllWindows()
