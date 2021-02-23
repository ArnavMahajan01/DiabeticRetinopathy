import numpy as np
import cv2


img = cv2.imread("/home/ayush/Documents/python_projects/semester_long_project/raw_dataset/training/images/21_training.tif")

# print(frame)

clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))

ig = img[:,:,1]
inormalized = cv2.equalizeHist(ig)
iClahe = clahe.apply(inormalized)

mask = cv2.VideoCapture("/home/ayush/Documents/python_projects/semester_long_project/raw_dataset/training/mask/21_training_mask.gif")

boolean, frame = mask.read()
frame = frame[:,:,1]


r = 0
while r < len(iClahe):
    c = 0
    while c < len(iClahe[0]):
        if(frame[r][c] < 20):
            iClahe[r][c] = frame[r][c]
        c = c+1
    r = r+1
        

flag = False
y1 = 0
y2 = 0
x1 = 0
x2 = 0

i = 0

for y in iClahe:
    for x in y:
        if(x > 20 and flag == False):
            flag = True
            y1 = i
            break
    if(flag == True):
        break
    i = i+1

flag = False
i = len(iClahe)-1

for y in iClahe[::-1]:
    for x in y:
        if(x > 20 and flag == False):
            flag = True
            y2 = i
            break
    if(flag == True):
        break
    i = i-1

flag = False
i = 0

while i < len(iClahe[0]):
    j = 0
    while j < len(iClahe):
        if(iClahe[j][i] > 20 and flag == False):
            x1 = i
            flag = True
            break
        j = j+1
    if(flag == True):
        break
    i = i+1

i = len(iClahe[0])-1
flag = False

while i >= 0:
    j = 0
    while j < len(iClahe):
        if(iClahe[j][i] > 20 and flag == False):
            x2 = i
            flag = True
            break
        j = j+1
    if(flag == True):
        break
    i = i-1


crop = iClahe[y1:y2, x1:x2]

cv2.imshow('final pre processed image', crop)


cv2.waitKey(0)




