#!/usr/bin/env python
# -*- coding: utf-8 -*-
from PIL import Image

import copy
import argparse

import cv2 as cv
import numpy as np
import mediapipe as mp
import pickle 

from utils import CvFpsCalc


#My Codes
def preProcess(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.equalizeHist(img)
    img = img / 255 
    return img 
        
pickle_in = open("model_trained_hand.p","rb")
model = pickle.load(pickle_in) 




#son


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--max_num_hands", type=int, default=2)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    parser.add_argument('--use_brect', action='store_true')

    args = parser.parse_args()

    return args


def main():
    # 引数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    max_num_hands = args.max_num_hands
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = args.use_brect

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    while True:
        display_fps = cvFpsCalc.get()

        # カメラキャプチャ #####################################################
        ret, image = cap.read()
        if not ret:
            break
        
        
        
        image2 = cv.flip(image, 1)  # ミラー表示
        debug_image = copy.deepcopy(image2)

        # 検出実施 #############################################################
        image2 = cv.cvtColor(image2, cv.COLOR_BGR2RGB)
        results = hands.process(image2)

        # 描画 ################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                
                # 手の平重心計算
                cx, cy = calc_palm_moment(debug_image, hand_landmarks)
                # 外接矩形の計算
                
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                #print(brect)
                
                left, top, right, bottom = brect
                print(left, top, right, bottom)
                denemeImage = Image.fromarray(debug_image, 'RGB')
                denemeImage = denemeImage.crop((left, top, right, bottom))
                denemeImage=np.asarray(denemeImage) #np.uint8()
                
                hsvim = cv.cvtColor(denemeImage, cv.COLOR_BGR2HSV)
                lower = np.array([0, 48, 80], dtype = "uint8")
                upper = np.array([20, 255, 255], dtype = "uint8")
                skinRegionHSV = cv.inRange(hsvim, lower, upper)
                #blurred = cv.blur(skinRegionHSV, ksize=(2,2))
                
                kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
                skinMask = cv.erode(skinRegionHSV, kernel, iterations = 2)
                skinMask = cv.dilate(skinRegionHSV, kernel, iterations = 2)
                
                skinMask = cv.GaussianBlur(skinMask, (3, 3), 0)
                skin = cv.bitwise_and(denemeImage, denemeImage, mask = skinMask)
                
                thresh_new = np.hstack([denemeImage, skin])
                cv.imshow("deneme",thresh_new)
                
                
                
                
                
                
                
                ret,thresh = cv.threshold(skinRegionHSV,0,255,cv.THRESH_BINARY)
                #_,thresh_img = cv.threshold(image,thresh=60,maxval=255,type=cv.THRESH_BINARY)
                cv.imshow("thresh", thresh)
                
                img = np.asarray(thresh)
                img = cv.resize(img, (120, 320)) # input shape
        
                img = cv.equalizeHist(img)
                img = img / 255 
                
                img = img.reshape(1,120, 320,1) # 1 adet resim, (120, 320) boytunda, channel = 1
                
                classIndex = int(model.predict_classes(img))

                predictions = model.predict(img)
                probVal = np.amax(predictions)
                    #print(classIndex, probVal)

                if probVal > 0.7:
                    if classIndex == 0:
                        cv.putText(image, "Palm"+"     " + str(probVal),(50,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv.LINE_AA)
                    elif classIndex == 1:
                        cv.putText(image, "L"+"     " + str(probVal),(50,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv.LINE_AA)
                    elif classIndex == 2:
                        cv.putText(image, "Fist"+"     " + str(probVal),(50,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv.LINE_AA)
                    elif classIndex == 3:
                        cv.putText(image, "Thumb"+"     " + str(probVal),(50,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv.LINE_AA)
                    elif classIndex == 4:
                        cv.putText(image, "Ok"+"     " + str(probVal),(50,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv.LINE_AA)
                    elif classIndex == 5:
                        cv.putText(image, "C"+"     " + str(probVal),(50,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv.LINE_AA)
        
                cv.imshow("Hand Detection", image)
                
                cv.imshow("DenemeImage",denemeImage)
                # 描画
                debug_image = draw_landmarks(debug_image, cx, cy,
                                             hand_landmarks, handedness)
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)

        cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)

        # キー処理(ESC：終了) #################################################
        
        # model eklenicek ve işlemler yapılacak ---> frame olarak image kullanılacak
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        
        
#        img = np.asarray(debug_image) # frameyi arraya çevirdik
#        img = cv.resize(img, (120, 320)) # input shape
#        img = preProcess(img)
#
#        img = img.reshape(1,120, 320,1) # 1 adet resim, (120, 320) boytunda, channel = 1
#        
#        hsvim = cv.cvtColor(debug_image, cv.COLOR_BGR2HSV)
#        lower = np.array([0, 48, 80], dtype = "uint8")
#        upper = np.array([20, 255, 255], dtype = "uint8")
#        skinRegionHSV = cv.inRange(hsvim, lower, upper)
#        #blurred = cv.blur(skinRegionHSV, ksize=(2,2))
#        ret,thresh = cv.threshold(skinRegionHSV,0,255,cv.THRESH_BINARY)
#        #_,thresh_img = cv.threshold(image,thresh=60,maxval=255,type=cv.THRESH_BINARY)
#        #cv.imshow("thresh", thresh)
#        
#        img = np.asarray(thresh)
#        img = cv.resize(img, (120, 320)) # input shape
#        
#        img = cv.equalizeHist(img)
#        img = img / 255 
#        
#        img = img.reshape(1,120, 320,1) # 1 adet resim, (120, 320) boytunda, channel = 1
#
#        #predict 
#        classIndex = int(model.predict_classes(img))
#
#        predictions = model.predict(img)
#        probVal = np.amax(predictions)
#        #print(classIndex, probVal)
#
#        if probVal > 0.7:
#            if classIndex == 0:
#                cv.putText(image, "Palm"+"     " + str(probVal),(50,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv.LINE_AA)
#            elif classIndex == 1:
#                cv.putText(image, "L"+"     " + str(probVal),(50,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv.LINE_AA)
#            elif classIndex == 2:
#                cv.putText(image, "Fist"+"     " + str(probVal),(50,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv.LINE_AA)
#            elif classIndex == 3:
#                cv.putText(image, "Thumb"+"     " + str(probVal),(50,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv.LINE_AA)
#            elif classIndex == 4:
#                cv.putText(image, "Ok"+"     " + str(probVal),(50,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv.LINE_AA)
#            elif classIndex == 5:
#                cv.putText(image, "C"+"     " + str(probVal),(50,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv.LINE_AA)
#        
#        cv.imshow("Hand Detection", image)
        
        
        
        
        
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #son
        
        
        
        key = cv.waitKey(1)
        
        if key == 27:  # ESC
            break

        # 画面反映 #############################################################
        cv.imshow('MediaPipe Hand Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


def calc_palm_moment(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    palm_array = np.empty((0, 2), int)

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        if index == 0:  # 手首1
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 1:  # 手首2
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 5:  # 人差指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 9:  # 中指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 13:  # 薬指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 17:  # 小指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
    M = cv.moments(palm_array)
    cx, cy = 0, 0
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

    return cx, cy


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def draw_landmarks(image, cx, cy, landmarks, handedness):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # キーポイント
    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append((landmark_x, landmark_y))

        if index == 0:  # 手首1
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 1:  # 手首2
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)

    # 接続線
    if len(landmark_point) > 0:
        # 親指
        cv.line(image, landmark_point[2], landmark_point[3], (0, 255, 0), 2)
        cv.line(image, landmark_point[3], landmark_point[4], (0, 255, 0), 2)

        # 人差指
        cv.line(image, landmark_point[5], landmark_point[6], (0, 255, 0), 2)
        cv.line(image, landmark_point[6], landmark_point[7], (0, 255, 0), 2)
        cv.line(image, landmark_point[7], landmark_point[8], (0, 255, 0), 2)

        # 中指
        cv.line(image, landmark_point[9], landmark_point[10], (0, 255, 0), 2)
        cv.line(image, landmark_point[10], landmark_point[11], (0, 255, 0), 2)
        cv.line(image, landmark_point[11], landmark_point[12], (0, 255, 0), 2)

        # 薬指
        cv.line(image, landmark_point[13], landmark_point[14], (0, 255, 0), 2)
        cv.line(image, landmark_point[14], landmark_point[15], (0, 255, 0), 2)
        cv.line(image, landmark_point[15], landmark_point[16], (0, 255, 0), 2)

        # 小指
        cv.line(image, landmark_point[17], landmark_point[18], (0, 255, 0), 2)
        cv.line(image, landmark_point[18], landmark_point[19], (0, 255, 0), 2)
        cv.line(image, landmark_point[19], landmark_point[20], (0, 255, 0), 2)

        # 手の平
        cv.line(image, landmark_point[0], landmark_point[1], (0, 255, 0), 2)
        cv.line(image, landmark_point[1], landmark_point[2], (0, 255, 0), 2)
        cv.line(image, landmark_point[2], landmark_point[5], (0, 255, 0), 2)
        cv.line(image, landmark_point[5], landmark_point[9], (0, 255, 0), 2)
        cv.line(image, landmark_point[9], landmark_point[13], (0, 255, 0), 2)
        cv.line(image, landmark_point[13], landmark_point[17], (0, 255, 0), 2)
        cv.line(image, landmark_point[17], landmark_point[0], (0, 255, 0), 2)

    # 重心 + 左右
    if len(landmark_point) > 0:
        # handedness.classification[0].index
        # handedness.classification[0].score

        cv.circle(image, (cx, cy), 12, (0, 255, 0), 2)
        cv.putText(image, handedness.classification[0].label[0],
                   (cx - 6, cy + 6), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                   2, cv.LINE_AA)  # label[0]:一文字目だけ

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 255, 0), 2)

    return image


if __name__ == '__main__':
    main()
