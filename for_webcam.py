from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2,time
from scipy.spatial import distance as dist
import math

cam_id = 0 #0 for internal camera provide any other id e.g 1,2 to use external camera
cap = cv2.VideoCapture(cam_id)
#body_cascade = cv2.CascadeClassifier('Harcascades/haarcascade_fullbody.xml')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
c = 0
while  True:
    ret , image = cap.read()
    image = cv2.resize(image , (700,700))
    orig = image.copy()
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(6, 6),padding=(14, 14), scale=1.05)
    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 1)
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    lst = [chr(i+65) for i in range(len(pick))]
    i = 0
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 1)
        cv2.putText(image, lst[i], (int((xA+xB)/2),yA), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 127, 50), 3)
        i+=1

    pick_x = []
    for i in pick:
        pick_x.append(list(i))
    #print(pick_x)
    people = []
    for i in range(len(pick_x)):
        for j in range(i,len(pick_x)):
            if pick_x[i]==pick_x[j]:
                pass
            else:
                fp = pick_x[i]
                sp = pick_x[j]
                fp_x1,fp_y1,fp_x2,fp_y2 = fp[0],fp[1],fp[2],fp[3]
                sp_x1,sp_y1,sp_x2,sp_y2 = sp[0],sp[1],sp[2],sp[3]
                midpoint_fp = int((fp_x1 + fp_x2)/2) , int((fp_y1 + fp_y2)/2)
                midpoint_sp = int((sp_x1 + sp_x2)/2) , int((sp_y1 + sp_y2)/2)
                cv2.line(image,midpoint_fp,midpoint_sp,(255,127,30),2)
                d = math.sqrt((midpoint_sp[0]-midpoint_fp[0])**2+(midpoint_sp[1]-midpoint_fp[1])**2)
                index_fp = pick_x.index(fp)
                index_sp = pick_x.index(sp)
                fpn = lst[index_fp]
                spn = lst[index_sp]
                print('Distance from '+fpn+' to '+spn,' is ' , round((d*0.021458)/1.7,2),'meters')
                cv2.putText(image, str(round((d*0.021458)/1.7,2)), (int((midpoint_fp[0]+midpoint_sp[0])/2) , int((midpoint_fp[1]+midpoint_sp[1])/2)) , cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 70, 255), 1)
    #cv2.imshow("Before NMS", orig)
    cv2.imshow("Final result", image)
    c+=1
    print('Frame = ',c)
    key = cv2.waitKey(33)
    if key==27:
        cv2.destroyAllWindows()
        break
    #cv2.waitKey(0)
cap.release()