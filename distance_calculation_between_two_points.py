import numpy as np
import cv2
from scipy.spatial import distance as dist
import math
'''
Demonstration of simple technique for calculating distance in an image
'''


img = np.zeros((1000,1000))
img[200][200] = 1
img[900][900] = 1
x1,y1,x2,y2 = 200,200,900,900
#100  100   400    400
#D = dist.euclidean((400, 100), (400, 100))
d = math.sqrt((x2-x1)**2+(y2-y1)**2)
print(d*0.021458, 'cm')

cv2.line(img,(200,200),(900,900),1,5)

cv2.putText(img, 'a', (200,200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)
cv2.putText(img, 'b', (900,900), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)


cv2.imshow('-',img)
cv2.waitKey(0)
