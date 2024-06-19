import mediapipe as mp
import cv2 
import numpy as np

mpHands = mp.solutions.hands
hands_model = mpHands.Hands(static_image_mode=True, 
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

def get_hands(img_path):
    # print(img_path)
    def increase_bbox(bbox, scale_factor):
        x, y, w, h = bbox
        delta_w = int((scale_factor - 1) * w / 2)
        delta_h = int((scale_factor - 1) * h / 2)
        return x - delta_w, y - delta_h, w + 2 * delta_w, h + 2 * delta_h

    img = cv2.imread(img_path)
    # img = cv2.flip(img, 1)
    # print(img)
    gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    # show_img(gray_img)
    img_h,img_w=gray_img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hand_imgs=[]
    results = hands_model.process(imgRGB)
    # print(len(results.multi_hand_landmarks))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            landmark_points = []
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * img.shape[1])
                y = int(landmark.y * img.shape[0])
                landmark_points.append([x, y])

            landmark_points = np.array(landmark_points)  
            x, y, w, h = cv2.boundingRect(landmark_points)
            # show_img(gray_img[y:y+h,x:x+w])
            
            scale_factor = 1.2
            x, y, w, h = increase_bbox((x, y, w, h), scale_factor)
            sx=max(0,x)
            sy=max(0,y)
            ly=min(img_h,h+y)
            lx=min(img_w,w+x)
            hand_imgs.append( gray_img[sy:ly,sx:lx])
            # show_img(hand_img)
    return hand_imgs

def get_hands_pos(frame):
    def increase_bbox(bbox, scale_factor):
        x, y, w, h = bbox
        delta_w = int((scale_factor - 1) * w / 2)
        delta_h = int((scale_factor - 1) * h / 2)
        return x - delta_w, y - delta_h, w + 2 * delta_w, h + 2 * delta_h
    img = frame.copy()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h,img_w=imgRGB.shape[:2]
    results = hands_model.process(imgRGB)
    # print(len(results.multi_hand_landmarks))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            landmark_points = []
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * img.shape[1])
                y = int(landmark.y * img.shape[0])
                landmark_points.append([x, y])

            landmark_points = np.array(landmark_points)  
            x, y, w, h = cv2.boundingRect(landmark_points)
            # show_img(gray_img[y:y+h,x:x+w])
            
            scale_factor = 1.2
            x, y, w, h = increase_bbox((x, y, w, h), scale_factor)
            sx=max(0,x)
            sy=max(0,y)
            ly=min(img_h,h+y)
            lx=min(img_w,w+x)
            return [sx,sy,lx,ly]