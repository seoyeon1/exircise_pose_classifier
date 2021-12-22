from tkinter import *
from tkinter.font import *
import cv2
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
import mediapipe as mp
import sklearn
import os
import pygame
pygame.mixer.init()

window = Tk()
window.title("AIoT") 

width = window.winfo_screenwidth() 
height = window.winfo_screenheight()


window.state('zoomed')
print(width)
print(height)

font_t = Font(family="맑은 고딕", size=40)
font_b = Font(family="맑은 고딕", size=30)
title = Label(window, text="홈트레이닝",font=font_t).pack(side="top")

#----------------------------------------------------------------------------------------#
# Function (Calculate angle)

def calculate_angle(a,b,c): # First, Mid, End keypoint
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

#----------------------------------------------------------------------------------------#
# 각 동작을 구성하는 관절 리스트

joint_list = {'Push_up' : [[11,13,15] , [11,23,25]],
             'Lunge' : [[24,26,28] , [23,25,27], [12,24,26], [11,23,25]],
             'L_fly' : [[11,13,15], [14,12,11], [12,14,16], [13,11,23], [14,12,24]],
             'Squat' : [[24,26,28] , [23,25,27]],
             'Dumb_bell' : [[11,13,15], [12,14,16], [14, 12, 24], [13, 11, 23]]}


# 칼로리 합계 변수
sum = 0

#----------------------------------------------------------------------------------------#
# 인공지능
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
#----------------------------------------------------------------------------------------#
# 비디오, 웹캠 시작
# 초급
def vod1_start():

    
    # 변수

    counter_1 = 0 #카운터
    stat = None #운동 상태
    angles = [] #각도 저장 리스트
    kcal = 0
    # *********비디오 파일********#
    vod_file = "lv1.mp4"
    # *********모델 파일********#
    with open('fitness_pose.pkl', 'rb') as f: #모델 로드
        model = pickle.load(f)
    
    vfile = cv2.VideoCapture(vod_file)
    cfile = cv2.VideoCapture(1)  # 노트북카메라 : 0, USB 카메라 : 1
    # pygame.mixer.music.load('cute.mp3')
    pygame.mixer.music.load('pretty.mp3')
    # pygame.mixer.music.play()
    pygame.mixer.music.play()
    
    with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
        while True:
            vret, frame_v = vfile.read()
            cret, frame_c = cfile.read()

            # current_time = time.time() - prev_time

            # 화면 사이즈 dsize로 조절<가로,세로>   
            frame_v = cv2.resize(frame_v ,dsize=(760,850), interpolation=cv2.INTER_AREA) 
            frame_c = cv2.resize(frame_c ,dsize=(760,850), interpolation=cv2.INTER_AREA) 


            # 웹캠이 켜지면 비디오를 킴 <웹캠이 비디오보다 늦게 켜짐>
            if cret:
                
                
                image = cv2.cvtColor(frame_c, cv2.COLOR_BGR2RGB)
                image = cv2.flip(image,1)
                image.flags.writeable = False
        
                results = holistic.process(image)
                
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
                # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 96, 66), thickness=3, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(111, 245, 66), thickness=4, circle_radius=3) 
                                 )


         
                try:
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
                    # Model Detection
                    X = pd.DataFrame([pose_row]) #class를 frame에 보여줌
                    fitness_pose_class = model.predict(X)[0] #클래스 예측
                    fitness_pose_prob = model.predict_proba(X)[0] # 확률예측
                    #print(fitness_pose_class, fitness_pose_prob)#예측한 동작, 정확도


                    # 각도 계산 & 카운팅
                    

                    if fitness_pose_class in joint_list:
                        
                        for joint in joint_list[fitness_pose_class]:#알고 싶은 손가락 각도를 이룬 렌드마크 별 좌표 값 추출(a,b,c)
                            a = np.array([pose[joint[0]].x, pose[joint[0]].y]) # First coord
                            b = np.array([pose[joint[1]].x, pose[joint[1]].y]) # Second coord
                            c = np.array([pose[joint[2]].x, pose[joint[2]].y]) # Third coord
                    # Calculate angle
                            angle = calculate_angle(a, b, c)
                    # Counter 구현
                            angles.append(angle)
                        
                        # PUSH_UP
                        
                        if fitness_pose_class == 'Push_up': 
                            if angles[0] > 160:
                                stat = 'down'
                                if angles[1] < 100:
                                    pygame.mixer.music.load('hip.mp3')
                                    pygame.mixer.music.play() 
                            if angles[0] < 50 and stat =='down':
                                stat="up"
                                counter_1 +=1

                                pygame.mixer.music.load('ok.mp3')
                                pygame.mixer.music.play()
                                if counter_1 !=0 and (counter_1 % 10 == 0):
                                    pygame.mixer.music.load('cheer.wav')
                                    pygame.mixer.music.play()
                                kcal += 5

                            angles.clear()
                        
                        # LUNGE
                                
                        elif fitness_pose_class == 'Lunge': 
                            if angles[0] >160 and angles[1]>160: 
                                stat = 'ready_L'
                            if stat == 'ready_L' and 130<angles[2]<150 and 130<angles[3]<150:
                                pygame.mixer.music.load('down.mp3')
                                pygame.mixer.music.play()
                            if angles[0]<130 and angles[1]<130 and stat == 'ready_L':
                                stat = 'lunge'
                                counter_1 +=1

                                pygame.mixer.music.load('ok.mp3')
                                pygame.mixer.music.play()
                                if counter_1 !=0 and (counter_1 % 10 == 0):
                                    pygame.mixer.music.load('cheer.wav')
                                    pygame.mixer.music.play()
                                kcal += 3

                            angles.clear()

                        # L_FLY
                            
                        elif fitness_pose_class == 'L_fly':
                            if 80 < angles[0] < 110 and angles[1] > 150 and 80 < angles[2] <110 :
                                stat = 'ready'
                            if stat == 'ready' and angles[3] < 70 and angles[4] < 70:
                                stat = 'bad'
                                pygame.mixer.music.load('arm.mp3')
                                pygame.mixer.music.play()

                            if stat == 'ready' and angles[1] < 90:
                                stat = 'lfly'
                                counter_1 += 1
                                kcal += 1
                                # print(counter, kcal)
                                pygame.mixer.music.load('ok.mp3')
                                pygame.mixer.music.play()
                                if counter_1 !=0 and (counter_1 % 10 == 0):
                                    pygame.mixer.music.load('cheer.wav')
                                    pygame.mixer.music.play()
                                # kcal += 1
                                # print(kcal)
                            
                            angles.clear()

                        # SQUAT

                        elif fitness_pose_class == 'Squat':
                            if angles[0] > 160 and angles[1] > 160:
                                stat = 'read'
                            if stat=='read' and 140 > angles[0] > 100 and 140 > angles[1] > 100:
                                pygame.mixer.music.load('down.mp3')
                                pygame.mixer.music.play()
                            if angles[0] < 100 and angles[1] < 100 and stat == 'read':
                                stat = 'squat'
                                counter_1 += 1
                                # print(counter)
                                pygame.mixer.music.load('ok.mp3')
                                pygame.mixer.music.play()
                                if counter_1 !=0 and (counter_1 % 10 == 0):
                                    pygame.mixer.music.load('cheer.wav')
                                    pygame.mixer.music.play()
                                kcal += 5
                                # print(kcal)
                            # elif stat == 'ready' and 120 > angles[0] > 90 and 120 > angles[1] > 90:
                            #     stat = 'bad'
                            #     pygame.mixer.music.load('down.mp3')
                            #     pygame.mixer.music.play()
                            angles.clear()

                        # DUMB_BELL

                        elif fitness_pose_class == 'Dumb_bell':
                            if angles[0] >160 and angles[1] > 160:
                                stat = 'ready_D'
                            if 80 < angles[2] < 100 and 80 < angles[3] < 100 and stat=='ready_D':
                                stat = 'dumb'
                                counter_1 += 1
                                # print(counter)
                                pygame.mixer.music.load('ok.mp3')
                                pygame.mixer.music.play()
                                if counter_1 !=0 and (counter_1 % 10 == 0):
                                    pygame.mixer.music.load('cheer.wav')
                                    pygame.mixer.music.play()
                                kcal += 2

                            angles.clear()








                    ############# Display Window ###################33

                    # status box
                    cv2.rectangle(image, (0,0),(250,80),(0,0,0),-1)



                    # Count
                    cv2.putText(image, 'Count', (15,20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(counter_1), 
                                (15,70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,255), 2, cv2.LINE_AA)
        
                    # Kcal
                    cv2.putText(image, 'kcal', (120,20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(kcal), 
                                (120,70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,255), 2, cv2.LINE_AA)
        
                except:
                    pass        
                
                cv2.imshow('VIDEO', frame_v)
                cv2.imshow('CAMERA', image)
                if cv2.waitKey(1)!=-1:
                    global sum 
                    sum += kcal
                    break
            else:
                break
        
    vfile.release()     
    cfile.release()                   
    cv2.destroyAllWindows()

# 중급
def vod2_start():
     # Curl counter 변수
    counter_2 = 0 #카운터
    stat = None #운동 상태
    angles = [] #각도 저장 리스트
    kcal = 0
    # *********비디오 파일********#
    vod_file = "lv2.mp4"
    # *********모델 파일********#
    with open('fitness_pose.pkl', 'rb') as f: #모델 로드
        model = pickle.load(f)
    
    vfile = cv2.VideoCapture(vod_file)
    cfile = cv2.VideoCapture(1)  # 노트북카메라 : 0, USB 카메라 : 1
    pygame.mixer.music.load('pretty.mp3')
    # pygame.mixer.music.play()
    pygame.mixer.music.play()
    
    with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
        while True:
            vret, frame_v = vfile.read()
            cret, frame_c = cfile.read()
    
            # 화면 사이즈 dsize로 조절<가로,세로>   
            frame_v = cv2.resize(frame_v ,dsize=(760,850), interpolation=cv2.INTER_AREA) 
            frame_c = cv2.resize(frame_c ,dsize=(760,850), interpolation=cv2.INTER_AREA) 
  
            # 웹캠이 켜지면 비디오를 킴 <웹캠이 비디오보다 늦게 켜짐>
            if cret:
                image = cv2.cvtColor(frame_c, cv2.COLOR_BGR2RGB)
                image = cv2.flip(image,1)
                image.flags.writeable = False
        
                results = holistic.process(image)
                
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
                # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 96, 66), thickness=3, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(111, 245, 66), thickness=4, circle_radius=3) 
                                 )


         
                try:
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
                    # Model Detection
                    X = pd.DataFrame([pose_row]) #class를 frame에 보여줌
                    fitness_pose_class = model.predict(X)[0] #클래스 예측
                    fitness_pose_prob = model.predict_proba(X)[0] # 확률예측
                    #print(fitness_pose_class, fitness_pose_prob)#예측한 동작, 정확도


                    # 각도 계산 & 카운팅
                    

                    if fitness_pose_class in joint_list:
                        # kcal = 0
                        for joint in joint_list[fitness_pose_class]:#알고 싶은 손가락 각도를 이룬 렌드마크 별 좌표 값 추출(a,b,c)
                            a = np.array([pose[joint[0]].x, pose[joint[0]].y]) # First coord
                            b = np.array([pose[joint[1]].x, pose[joint[1]].y]) # Second coord
                            c = np.array([pose[joint[2]].x, pose[joint[2]].y]) # Third coord
                    # Calculate angle
                            angle = calculate_angle(a, b, c)
                    # Curl Counter 구현
                            angles.append(angle)
                        
                        if fitness_pose_class == 'Push_up':
                            if angles[0] > 160:
                                stat = 'down'
                                if angles[1] < 100:
                                    pygame.mixer.music.load('hip.mp3')
                                    pygame.mixer.music.play() 
                            if angles[0] < 50 and stat =='down':
                                stat="up"
                                counter_2 +=1
                                # print(counter)
                                pygame.mixer.music.load('ok.mp3')
                                pygame.mixer.music.play()
                                if counter_2 !=0 and (counter_2 % 10 == 0):
                                    pygame.mixer.music.load('cheer.wav')
                                    pygame.mixer.music.play()
                                kcal += 5
                                # print(kcal)
                            angles.clear()
                                
                        elif fitness_pose_class == 'Lunge':
                            if angles[0] >160 and angles[1]>160: 
                                stat = 'ready_L'
                            if stat == 'ready_L' and 120<angles[2]<150 and 120<angles[3]<150:
                                pygame.mixer.music.load('down.mp3')
                                pygame.mixer.music.play()
                            if angles[0]<120 and angles[1]<120 and stat == 'ready_L':
                                stat = 'lunge'
                                counter_2 +=1
                                # print(counter)
                                pygame.mixer.music.load('ok.mp3')
                                pygame.mixer.music.play()
                                if counter_2 !=0 and (counter_2 % 10 == 0):
                                    pygame.mixer.music.load('cheer.wav')
                                    pygame.mixer.music.play()
                                kcal += 3
                                # print(kcal)
                            angles.clear()
                            
                        elif fitness_pose_class == 'L_fly':
                            if 80 < angles[0] < 110 and angles[1] > 150 and 80 < angles[2] <110 :
                                stat = 'ready'
                            if stat == 'ready' and angles[3] < 70 and angles[4] < 70:
                                stat = 'bad'
                                pygame.mixer.music.load('arm.mp3')
                                pygame.mixer.music.play()

                            if stat == 'ready' and angles[1] < 90:
                                stat = 'lfly'
                                counter_2 += 1
                                kcal += 1
                                # print(counter, kcal)
                                pygame.mixer.music.load('ok.mp3')
                                pygame.mixer.music.play()
                                if counter_2 !=0 and (counter_2 % 10 == 0):
                                    pygame.mixer.music.load('cheer.wav')
                                    pygame.mixer.music.play()
                                # kcal += 1
                                # print(kcal)
                            
                            angles.clear()

                        elif fitness_pose_class == 'Squat':
                            if angles[0] > 160 and angles[1] > 160:
                                stat = 'read'
                            if stat=='read' and 140 > angles[0] > 100 and 140 > angles[1] > 100:
                                pygame.mixer.music.load('down.mp3')
                                pygame.mixer.music.play()
                            if angles[0] < 100 and angles[1] < 100 and stat == 'read':
                                stat = 'squat'
                                counter_2 += 1
                                # print(counter)
                                pygame.mixer.music.load('ok.mp3')
                                pygame.mixer.music.play()
                                if counter_2 !=0 and (counter_2 % 10 == 0):
                                    pygame.mixer.music.load('cheer.wav')
                                    pygame.mixer.music.play()
                                kcal += 5
                                # print(kcal)
                            # elif stat == 'ready' and 120 > angles[0] > 90 and 120 > angles[1] > 90:
                            #     stat = 'bad'
                            #     pygame.mixer.music.load('down.mp3')
                            #     pygame.mixer.music.play()
                            angles.clear()







                                        ############# Display ###################33

                    # status box
                    cv2.rectangle(image, (0,0),(250,80),(0,0,0),-1)



                    # Count
                    cv2.putText(image, 'Count', (15,20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(counter_2), 
                                (15,70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,255), 2, cv2.LINE_AA)
        
                    # Kcal
                    cv2.putText(image, 'kcal', (120,20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(kcal), 
                                (120,70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,255), 2, cv2.LINE_AA)
                except:
                    pass        
                
                cv2.imshow('VIDEO', frame_v)
                cv2.imshow('CAMERA', image)
                if cv2.waitKey(1)!=-1:
                    global sum 
                    sum += kcal
                    break
            else:
                break
        
    vfile.release()     
    cfile.release()                   
    cv2.destroyAllWindows()

# 고급
def vod3_start():
     # Curl counter 변수
    counter = 0 #카운터
    stat = None #운동 상태
    angles = [] #각도 저장 리스트
    kcal = 0
    # *********비디오 파일********#
    vod_file = "lv3.mp4"
    # *********모델 파일********#
    with open('fitness_pose.pkl', 'rb') as f: #모델 로드
        model = pickle.load(f)
    
    vfile = cv2.VideoCapture(vod_file)
    cfile = cv2.VideoCapture(1)  # 노트북카메라 : 0, USB 카메라 : 1
    pygame.mixer.music.load('cute.mp3')
    # pygame.mixer.music.play()
    pygame.mixer.music.play()
    
    with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
        while True:
            vret, frame_v = vfile.read()
            cret, frame_c = cfile.read()
    
            # 화면 사이즈 dsize로 조절<가로,세로>   
            frame_v = cv2.resize(frame_v ,dsize=(760,850), interpolation=cv2.INTER_AREA) 
            frame_c = cv2.resize(frame_c ,dsize=(760,850), interpolation=cv2.INTER_AREA) 
  
            # 웹캠이 켜지면 비디오를 킴 <웹캠이 비디오보다 늦게 켜짐>
            if cret:
                image = cv2.cvtColor(frame_c, cv2.COLOR_BGR2RGB)
                image = cv2.flip(image,1)
                image.flags.writeable = False
        
                results = holistic.process(image)
                
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
                # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 96, 66), thickness=3, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(111, 245, 66), thickness=4, circle_radius=3) 
                                 )


         
                try:
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
                    # Model Detection
                    X = pd.DataFrame([pose_row]) #class를 frame에 보여줌
                    fitness_pose_class = model.predict(X)[0] #클래스 예측
                    fitness_pose_prob = model.predict_proba(X)[0] # 확률예측
                    #print(fitness_pose_class, fitness_pose_prob)#예측한 동작, 정확도


                    # 각도 계산 & 카운팅
                    

                    if fitness_pose_class in joint_list:
                        # kcal = 0
                        for joint in joint_list[fitness_pose_class]:#알고 싶은 손가락 각도를 이룬 렌드마크 별 좌표 값 추출(a,b,c)
                            a = np.array([pose[joint[0]].x, pose[joint[0]].y]) # First coord
                            b = np.array([pose[joint[1]].x, pose[joint[1]].y]) # Second coord
                            c = np.array([pose[joint[2]].x, pose[joint[2]].y]) # Third coord
                    # Calculate angle
                            angle = calculate_angle(a, b, c)
                    # Curl Counter 구현
                            angles.append(angle)
                        
                        if fitness_pose_class == 'Push_up':
                            if angles[0] > 160:
                                stat = 'down'
                                if angles[1] < 100:
                                    pygame.mixer.music.load('hip.mp3')
                                    pygame.mixer.music.play() 
                            if angles[0] < 50 and stat =='down':
                                stat="up"
                                counter +=1
                                # print(counter)
                                pygame.mixer.music.load('ok.mp3')
                                pygame.mixer.music.play()
                                if counter !=0 and (counter % 10 == 0):
                                    pygame.mixer.music.load('cheer.wav')
                                    pygame.mixer.music.play()
                                kcal += 5
                                # print(kcal)
                            angles.clear()
                                
                        elif fitness_pose_class == 'Lunge':
                            if angles[0] >160 and angles[1]>160: 
                                stat = 'ready_L'
                            if stat == 'ready_L' and 120<angles[2]<150 and 120<angles[3]<150:
                                pygame.mixer.music.load('down.mp3')
                                pygame.mixer.music.play()
                            if angles[0]<120 and angles[1]<120 and stat == 'ready_L':
                                stat = 'lunge'
                                counter +=1
                                # print(counter)
                                pygame.mixer.music.load('ok.mp3')
                                pygame.mixer.music.play()
                                if counter !=0 and (counter % 10 == 0):
                                    pygame.mixer.music.load('cheer.wav')
                                    pygame.mixer.music.play()
                                kcal += 3
                                # print(kcal)
                            angles.clear()
                            
                        elif fitness_pose_class == 'L_fly':
                            if 80 < angles[0] < 110 and angles[1] > 150 and 80 < angles[2] <110 :
                                stat = 'ready'
                            if stat == 'ready' and angles[3] < 70 and angles[4] < 70:
                                stat = 'bad'
                                pygame.mixer.music.load('arm.mp3')
                                pygame.mixer.music.play()

                            if stat == 'ready' and angles[1] < 90:
                                stat = 'lfly'
                                counter += 1
                                kcal += 1
                                # print(counter, kcal)
                                pygame.mixer.music.load('ok.mp3')
                                pygame.mixer.music.play()
                                if counter !=0 and (counter % 10 == 0):
                                    pygame.mixer.music.load('cheer.wav')
                                    pygame.mixer.music.play()
                                # kcal += 1
                                # print(kcal)
                            
                            angles.clear()

                        elif fitness_pose_class == 'Squat':
                            if angles[0] > 160 and angles[1] > 160:
                                stat = 'read'
                            if stat=='read' and 140 > angles[0] > 100 and 140 > angles[1] > 100:
                                pygame.mixer.music.load('down.mp3')
                                pygame.mixer.music.play()
                            if angles[0] < 100 and angles[1] < 100 and stat == 'read':
                                stat = 'squat'
                                counter += 1
                                # print(counter)
                                pygame.mixer.music.load('ok.mp3')
                                pygame.mixer.music.play()
                                if counter !=0 and (counter % 10 == 0):
                                    pygame.mixer.music.load('cheer.wav')
                                    pygame.mixer.music.play()
                                kcal += 5
                                # print(kcal)
                            # elif stat == 'ready' and 120 > angles[0] > 90 and 120 > angles[1] > 90:
                            #     stat = 'bad'
                            #     pygame.mixer.music.load('down.mp3')
                            #     pygame.mixer.music.play()
                            angles.clear()







                                        ############# Display ###################33

                    # status box
                    cv2.rectangle(image, (0,0),(250,80),(0,0,0),-1)



                    # Count
                    cv2.putText(image, 'Count', (15,20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(counter), 
                                (15,70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,255), 2, cv2.LINE_AA)
        
                    # Kcal
                    cv2.putText(image, 'kcal', (120,20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(kcal), 
                                (120,70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,255), 2, cv2.LINE_AA)
                except:
                    pass        
                
                cv2.imshow('VIDEO', frame_v)
                cv2.imshow('CAMERA', image)
                if cv2.waitKey(1)!=-1:
                    global sum 
                    sum += kcal
                    break
            else:
                break
        
    vfile.release()     
    cfile.release()                   
    cv2.destroyAllWindows()

# 비디오 끝
#----------------------------------------------------------------------------------------#

def window_en():
    print("칼로리 전송")
    window.destroy()

# 데이터베이스 서버        
# 종료버튼 누르면 데이터서버에 칼로리 전송

def window_end():
    db_url = 'https://home-training-d2c75-default-rtdb.firebaseio.com/'
    # ******서비스 계정의 비공개 키 파일이름 **********#
    cred = credentials.Certificate("home-training-d2c75.json")

    default_app = firebase_admin.initialize_app(cred, {'databaseURL':db_url})

    ref = db.reference()
    # calorie = 1000
    calorie = sum
    data = str(calorie) + "kcal"
    print(calorie)
    ref.update({datetime.today().strftime('%Y-%m-%d'):data})
    print("칼로리 전송 완료")
    window.destroy()    
    
rgt = Frame(window).pack(side='right')
lgt = Frame(window).pack(side='left')
   
select = Frame(window, pady=20)
select.pack()

# *******운동선택 이미지******** #
img1=PhotoImage(file="IMG1.png")
img2=PhotoImage(file="IMG2.png")
img3=PhotoImage(file="IMG3.png")

# 운동선택 버튼
# lV1 = Button(select, image=img1, command=vod1_start).grid(row=0, column=0, padx = 30, pady= 20)  
# lV2 = Button(select, image=img2, command=vod2_start).grid(row=0, column=1, padx = 30, pady= 20)
# lV3 = Button(select, image=img3, command=vod3_start).grid(row=0, column=2, padx = 30, pady= 20)

lV1 = Button(select, image=img1, command=vod1_start).grid(row=0, column=0)  
lV2 = Button(select, image=img2, command=vod2_start).grid(row=0, column=1)
lV3 = Button(select, image=img3, command=vod3_start).grid(row=0, column=2)

# 종료 버튼 누르면 칼로리 안드로이드에 소켓통신  
# ht_end = Button(window, text = "종료", font = font_b, width = 10, height= 1, background= 'orange',  command=window_end).pack(side="bottom", pady=30) 

ht_end = Button(window, text = "종료", font = font_b, width = 10, height= 1, background= 'orange',  command=window_end).pack(side="bottom", pady=10) 

window.mainloop()

