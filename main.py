import numpy as np
import cv2 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import dump
import joblib
import random
import pulsectl
import math
import csv
import sys
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from ROI import get_hands,get_hands_pos


        
def histogram(gradients,magnitude,div=4):
    h,w=gradients.shape
    hog=np.zeros((h//div,w//div,8))
    for i in range(0,h,div):
        for j in range(0,w,div):
            if(i+div>=h or j+div>=w):
                continue
            final_bin=np.zeros(8)
            for p in range(div):
                for q in range(div):
                    final_bin[int((gradients[i+p,j+q]%360)//45)]+=magnitude[i+p,j+q]
                    hog[i//div,j//div]=final_bin
    return hog

def hog_features(img,for_viz=False):
    img=cv2.resize(img,(150,250))
    gauss_img=cv2.GaussianBlur(img, (0, 0), sigmaX=1.2, sigmaY=1.2)
    # cv2.imwrite("gauss.png",gauss_img)
    x_conv=np.array([[-1,0,1]])
    dx=cv2.filter2D(gauss_img,-1,x_conv)
    dy=cv2.filter2D(gauss_img,-1,x_conv.T)
    gradients=np.degrees(np.arctan2(dy,dx))
    gradients[np.isnan(gradients)]=0
    gradients[gradients<0]+=360
    magnitude=np.sqrt((dx**2)+(dy**2))
    hog_arr=histogram(gradients,magnitude).reshape(-1)
    if(for_viz):
        hog_arr=histogram(gradients,magnitude,div=8)
    return hog_arr

def gen_non_hand_data(p1,save_to=None,flag=0,slide=100):
    count=0
    hog_data=[]
    img_list=os.listdir(p1)
    # print(len(img_list))
    for img_name in img_list:
        pth=os.path.join(p1,img_name)
        img=cv2.cvtColor(cv2.imread(pth), cv2.COLOR_BGR2GRAY)
        h,w=img.shape
        for i in range(0,h,slide):
            for j in range(0,w,slide):
                if(i+250>=h or j+150>=w ):
                    continue
                hog_data.append(hog_features(img[i:i+250,j:j+150]),for_viz=False)
                if(flag==1):
                    pth=os.path.join(save_to,str(count)+".jpg")
                    cv2.imwrite(pth,img[i:i+250,j:j+150])
                    count+=1
    return hog_data
    

def gen_hand_data_hand_detection(p1,save_to=None,train_size=(150,250),augmentation=True,data_ratio=0.2):
    count=0
    hog_data=[]
    imgs_list=os.listdir(p1)
    imgs_list=random.sample(imgs_list,int(len(imgs_list)*data_ratio))
    for name in imgs_list:
        path=os.path.join(p1,name)
        hands=get_hands(path)
        # print("tot handes given by function ",len(hands))
        all_hands=hands.copy()
        if(augmentation):
            for i in hands:
                h_flip=cv2.flip(i,0)
                v_flip=cv2.flip(i,1)
                all_hands.append(cv2.rotate(h_flip, cv2.ROTATE_90_CLOCKWISE))
                all_hands.append(cv2.rotate(h_flip, cv2.ROTATE_90_COUNTERCLOCKWISE))
                all_hands.append(cv2.rotate(i, cv2.ROTATE_90_CLOCKWISE))
                all_hands.append(cv2.rotate(i, cv2.ROTATE_90_COUNTERCLOCKWISE))
                all_hands.append(cv2.rotate(v_flip, cv2.ROTATE_90_CLOCKWISE))
                all_hands.append(cv2.rotate(v_flip, cv2.ROTATE_90_COUNTERCLOCKWISE))
                all_hands.append(v_flip)
                all_hands.append(h_flip)
            hands.clear()            
        for j in all_hands:
            if(j is None or j.shape[0]==0 or j.shape[1]==0):
                continue
            hog_data.append(hog_features(j,for_viz=False))
            if(save_to is not None):
                p2=os.path.join(save_to,str(count)+".jpg")
                h=cv2.resize(j,train_size)
                cv2.imwrite(p2,h)
                count+=1
        all_hands.clear()
    return hog_data


def gen_data_for_hand_detection(true_list_paths,flase_list_paths,save_to=None,augmentation=False,model_type=1,data_ratio=0.2):
    x=[]
    for path in true_list_paths:
        x.extend(gen_hand_data_hand_detection(path,save_to=save_to,augmentation=augmentation,data_ratio=data_ratio))
    true_count=len(x)
    print("true class generated!!")
    for path in flase_list_paths:
        if(model_type==0):
            x.extend(gen_non_hand_data(path,slide=100))
        else:
            x.extend(gen_hand_data_hand_detection(path,augmentation=augmentation,save_to=save_to,data_ratio=data_ratio))
    print("false class generated")
    false_count=len(x)-true_count
    x=np.array(x)
    y=[1]*true_count
    y.extend([0]*false_count)
    y=np.array(y)
    print(f"x len : {len(x)} , y len : {len(y)}")
    print(f"class 1: {true_count} class 0: {false_count}")
    num_samples = len(x)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    shuffled_x = x[indices]
    shuffled_y = y[indices]
    return shuffled_x,shuffled_y
    
def predict(path_t,path_f,classifier):
    hog_data=[]
    y_orig=[]
    for p1 in path_t:
        list_images=os.listdir(p1)
        for name in list_images:
            path=os.path.join(p1,name)
            hands=get_hands(path)
            for img in hands:
                hog_data.append(hog_features(img,for_viz=False))
    y_orig=[1]*len(hog_data)
    for p1 in path_f:
        list_images=os.listdir(p1)
        for name in list_images:
            path=os.path.join(p1,name)
            hands=get_hands(path)
            for img in hands:
                hog_data.append(hog_features(img,for_viz=False))
    y_orig.extend([0]*(len(hog_data)-len(y_orig)))
    
    y_pred=classifier.predict(hog_data)
    print("accuracy_score : ",accuracy_score(y_orig,y_pred))
    print(classification_report(y_orig, y_pred, target_names=['closed', 'open']))

    y_scores = classifier.predict_proba(hog_data)[:, 1] 
    fpr, tpr, thresholds = roc_curve(y_orig, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    #############################
    cm = confusion_matrix(y_orig, y_pred)

    # Extract TP, TN, FP, FN
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    print(f"Confusion Matrix:\n{cm}")
    print(f"True Positives (TP): {TP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    # return hog_data,y_pred,y_orig

def predict_imgs_inpath(path1,write_to_path,classifier):
    # print("infun")
    hog_data=[]
    list_images=os.listdir(path1)
    for name in list_images:
        print(name)
        path=os.path.join(path1,name)
        # print(path)
        hands=get_hands(path)
        for img in hands:
            hog_data.append(hog_features(img,for_viz=False))
    
    y_pred=classifier.predict(hog_data)
    y_pred_mapped = ['Close' if pred == 0 else 'Open' for pred in y_pred]

    predictions = []
    pred_index = 0
    for name in list_images:
        image_path = os.path.join(path1, name)
        hands = get_hands(image_path) 
        for _ in hands:
            predictions.append((image_path, y_pred_mapped[pred_index]))
            pred_index += 1
    csv_file_path = os.path.join(write_to_path, "predictions.csv")
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image Path', 'Predicted Class'])
        writer.writerows(predictions)


def train(true_list_paths,flase_list_paths,save_to=None,augmentation=False,model_type=1,data_ratio=0.2):
    x,y=gen_data_for_hand_detection(true_list_paths,flase_list_paths,save_to=save_to,augmentation=augmentation,model_type=model_type,data_ratio=data_ratio)
    svm_classifier = SVC(kernel='linear',probability=True)
    svm_classifier.fit(x, y)
    dump(svm_classifier, 'svm_model.joblib')
    print("svm trained!!")
    return svm_classifier


def viz(p):
    l=get_hands(p)
    for z,img in enumerate(l):
        hog_f=hog_features(img,for_viz=True)
        print(hog_f.shape)
        h,w,d=hog_f.shape
        nh,nw=h*d,w*d
        v_img=np.zeros((nh, nw), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                my=i*d+4
                mx=j*d+4
                ang=22.5
                scaled_fea=(hog_f[i,j]/np.max(hog_f[i,j]))*(d/2)
                scaled_fea = np.nan_to_num(scaled_fea, nan=0)
                for p in range(d):
                    if(hog_f[i,j,p]<200):
                        continue
                    end_x = mx + int(scaled_fea[p] * math.sin(math.radians((ang*(p+1))+90)))
                    end_y = my + int(scaled_fea[p] * math.cos(math.radians((ang*(p+1))+90)))
                    cv2.arrowedLine(v_img, (mx, my), (end_x, end_y), 255, thickness=1)  # Arrow color is set to 255 for white
        # plt.imshow(v_img,cmap="gray")
        # plt.plot()
        name="hog_img_viz"+str(z)+".png"
        cv2.imwrite(name,v_img)

def prob_hand_o_c(img,svm):
    x=hog_features(img).reshape(1,-1)
    # print(x.shape)
    y=svm.predict_proba(x)
    print(y[0][1])
    return y[0][1]

def set_system_volume(volume):
    with pulsectl.Pulse('set-volume') as pulse:
        for sink in pulse.sink_list():
            pulse.volume_set_all_chans(sink, volume)

def capture_frames(svm):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        box=get_hands_pos(frame)
        if box is not None:
            print("in box")
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            framecpy=frame.copy()
            gray_img=cv2.cvtColor(framecpy, cv2.COLOR_BGR2GRAY)
            prob=float(prob_hand_o_c(gray_img[box[1]:box[3], box[0]: box[2]],svm))
            print("volume : ",prob)
            set_system_volume(prob)
        cv2.imshow('Live Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



def scores(svm):
    close=["/home/prabhath/Desktop/sem2/CV_COL_780/Assignment_3/Final_dataset/closed1/valid","/home/prabhath/Desktop/sem2/CV_COL_780/Assignment_3/Final_dataset/closed2/valid","/home/prabhath/Desktop/sem2/CV_COL_780/Assignment_3/Final_dataset/closed3/valid"]
    open=["/home/prabhath/Desktop/sem2/CV_COL_780/Assignment_3/Final_dataset/open1/valid","/home/prabhath/Desktop/sem2/CV_COL_780/Assignment_3/Final_dataset/open2/valid"]
    hog_data,y_pred,y_orig=predict(open,close,svm)
    return hog_data,y_pred,y_orig
def main():
    do=int(sys.argv[1])
    print("do",do)
    if(do==2):  #model 1
        close_folder=sys.argv[3]
        open_folder=sys.argv[2]
        l1=os.listdir(close_folder)
        l2=os.listdir(open_folder)
        open=[]
        close=[]
        for i in range(len(l1)):
            close.append(os.path.join(close_folder,l1[i]))
        for i in range(len(l2)):
            open.append(os.path.join(open_folder,l2[i]))
        svm=train(open,close,data_ratio=0.3,model_type=1)
        # f=["/home/prabhath/Desktop/sem2/CV_COL_780/Assignment_3/Final_dataset/closed1/valid"]
        # t=["/home/prabhath/Desktop/sem2/CV_COL_780/Assignment_3/Final_dataset/open1/valid"]
        # hog_data,y_pred,y_orig=predict(t,f,svm)

    if(do==1): ###model 0
        hand_folder=sys.argv[2]
        non_hand_folder=sys.argv[3]
        l1=os.listdir(hand_folder)
        l2=os.listdir(non_hand_folder)
        hand=[]
        non_hand=[]
        for i in range(len(l1)):
            hand.append(os.path.join(hand_folder,l1[i]))
        for i in range(len(l2)):
            non_hand.append(os.path.join(non_hand_folder,l2[i]))
        svm=train(hand,non_hand,data_ratio=0.3,model_type=0)
        # t=["/home/prabhath/Desktop/sem2/CV_COL_780/Assignment_3/Final_dataset/closed1/valid"]
        # f=[]
        # hog_data,y_pred,y_orig=predict(f,t,svm)
    if(do==3): #volume control
        pth=sys.argv[2]
        svm = joblib.load(pth)
        capture_frames(svm)
    if(do==4): # images pred in given path and gen csv
        pth=sys.argv[4]
        svm = joblib.load(pth)
        test_data_path=sys.argv[2]
        save_to=sys.argv[3]
        predict_imgs_inpath(test_data_path,save_to,svm)

    if(do==5): ### test and plot curves
        pth=sys.argv[4]
        svm = joblib.load(pth)
        open_hand=sys.argv[2]
        closed_hand=sys.argv[3]
        l1=os.listdir(open_hand)
        l2=os.listdir(closed_hand)
        op=[]
        cl=[]
        for i in range(len(l1)):
            op.append(os.path.join(open_hand,l1[i]))
        for i in range(len(l2)):
            cl.append(os.path.join(closed_hand,l2[i]))
        predict(op,cl,svm)
    if(do==6):  ####img viz
        p=sys.argv[2]
        viz(p)

if __name__ == "__main__":
    main()