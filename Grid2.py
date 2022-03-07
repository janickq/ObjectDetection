import cv2
from imutils import contours
import numpy as np

# Load image, grayscale, and adaptive threshold
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

templates = [ ["Images2/Red.jpg",  "Red"],
                 ["Images2/Blue.jpg", "Blue"],
                 ["Images2/Yellow.jpg", "Yellow"],
                 ["Images2/Gurney.jpg", "Gurney"]]



blurSize = (5,5)


numOfTemplate = len(templates)
    
def template(img):
    img = cv2.blur(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), blurSize)
    imghist = cv2.calcHist([img],[0],None,[256],[0,256])
    cv2.normalize(imghist, imghist, 0, 1, cv2.NORM_MINMAX)
    for i in range(numOfTemplate):
        path = cv2.blur(cv2.imread(templates[i][0]), blurSize)
        temp = cv2.cvtColor(path, cv2.COLOR_BGR2HSV)
        temphist = cv2.calcHist([temp],[0],None,[256],[0,256])
        cv2.normalize(temphist, temphist, 0, 1, cv2.NORM_MINMAX)
        threshold = cv2.compareHist(imghist, temphist, cv2.HISTCMP_CORREL)
        color = templates[i][1]
        if threshold > 0.5:
            return threshold, color
        

def gethistogram(img):
    for i in enumerate([0 , 1, 2]):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    return hist
        


while True:
        
    frame, image = cam.read()
    imagecopy = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,57,5)

    # Filter out all numbers and noise to isolate only boxes
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 2000:
            cv2.drawContours(thresh, [c], -1, (0,0,0), -1)

    # Fix horizontal and vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel, iterations=9)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=4)

    # Sort by top to bottom and each row by left to right
    invert = 255 - thresh
    cnts = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")

    board_rows = []
    row = []
    for (i, c) in enumerate(cnts, 1):
        area = cv2.contourArea(c+1)
        if area < 4000:
            row.append(c)
             
            (cnts, _) = contours.sort_contours(row, method="left-to-right")
            board_rows.append(cnts)
            row = [] 

    # Iterate through each box
    
    for row in board_rows:
        for c in row:
            # mask = np.zeros(image.shape, dtype=np.uint8)
            # cv2.drawContours(mask, [c], -1, (255,255,255), -1)
            # result = cv2.bitwise_and(image, mask)
            # cv2.imshow("r",result)
            # result[mask==0] = 255
            x,y,w,h = cv2.boundingRect(c)
            result = imagecopy[y:y+h, x:x+w]
            cv2.imshow('result', result)
            print(template(result))
            cv2.waitKey(100)
    
    cv2.imshow('thresh', thresh)
    cv2.imshow('invert', invert)
    cv2.imshow('image', image)
    # cv2.waitKey()
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
        break