import cv2
import numpy as np
import imutils
from imutils import contours



cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def findcontour(img):

    imgcopy = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(imgGray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 5)

    # finding the border
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    c = 0
    for i in cnts:
            area = cv2.contourArea(i)
            if area > 25000:
                    if area > max_area:
                        max_area = area
                        best_cnt = i
                        img = cv2.drawContours(img, cnts, c, (0, 255, 0), 3)
            c+=1
    
    # mask image
    mask = np.zeros((imgGray.shape),np.uint8)
    cv2.drawContours(mask,[best_cnt],0,255,-1)
    cv2.drawContours(mask,[best_cnt],0,0,2)
    # cv2.imshow("mask", mask)

    # crop image
    out = np.zeros_like(imgGray)
    out[mask == 255] = imgGray[mask == 255]

    # threshold new image
    blur = cv2.GaussianBlur(out, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 1000:
            cv2.drawContours(thresh, [c], -1, (0,0,0), -1)
           
    
    

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel, iterations=9)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=4)
    
    # # Sort by top to bottom and each row by left to right
    thresh = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    invert = 255 - thresh
    cnts = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")
    cv2.imshow('thresh', thresh)
    cv2.imshow('invert', invert)
    sudoku_rows = []
    row = []
    for (i, c) in enumerate(cnts, 1):
        area = cv2.contourArea(c)
        if area < 50000:
            row.append(c)
            if i % 9 == 0:  
                (cnts, _) = contours.sort_contours(row, method="left-to-right")
                sudoku_rows.append(cnts)
                row = []

    # Iterate through each box
    for row in sudoku_rows:
        for c in row:
            mask = np.zeros(imgcopy.shape, dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, (255,255,255), -1)
            result = cv2.bitwise_and(imgcopy, mask)
            result[mask==0] = 255
            cv2.imshow('result', result)
            cv2.waitKey(175)

    

while True:
    
    frame_got, img = cam.read()
    if frame_got is False:
        break


    findcontour(img)
    # cv2.imshow("imgcopy", imgcopy)
  
    # cv2.imshow("img", img)
    # cv2.imshow('thresh', thresh)



    if cv2.waitKey(10) == 27:
        cv2.destroyAllWindows()
        break