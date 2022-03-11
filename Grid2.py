import cv2
from imutils import contours
import numpy as np
import argparse


# Load image, grayscale, and adaptive threshold
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# src = cv2.imread("Images2\image8.jpg")
# src = cv2.resize(src, (640, 480))

rows, cols = (6, 6)
 
boardArray = [[1]*cols]*rows


# # Pi 
#                Filename            BGR           text
templateInfoList = [ ["Images2/Red.jpg",  (0, 0, 255),  "Red"],
                 ["Images2/Blue.jpg", (255, 0, 0), "Blue"],
                 ["Images2/Yellow.jpg", (0, 255, 255), "Yellow"],
                 ["Images2/Yellow90.jpg", (0, 255, 255), "Yellow"],
                 ["Images2/Yellow180.jpg", (0, 255, 255), "Yellow"],
                 ["Images2/Yellow270.jpg", (0, 255, 255), "Yellow"],
                 ["Images2/Gurney.jpg", (0, 0, 0), "Gurney"],
                 ["Images2/Gurney90.jpg", (0, 0, 0), "Gurney"],
                 ["Images2/Gurney180.jpg", (0, 0, 0), "Gurney"],
                 ["Images2/Gurney270.jpg", (0, 0, 0), "Gurney"] ]

#  windows
#                Filename            BGR           text
# templateInfoList = [ ["Images2\Red.jpg",  (0, 0, 255),  "Red"],
#                  ["Images2\Blue.jpg", (255, 0, 0), "Blue"],
#                  ["Images2\Yellow.jpg", (0, 255, 255), "Yellow"],
#                  ["Images2\Yellow90.jpg", (0, 255, 255), "Yellow"],
#                  ["Images2\Yellow180.jpg", (0, 255, 255), "Yellow"],
#                  ["Images2\Yellow270.jpg", (0, 255, 255), "Yellow"],
#                  ["Images2\Gurney.jpg", (0, 0, 0), "Gurney"],
#                  ["Images2\Gurney90.jpg", (0, 0, 0), "Gurney"],
#                  ["Images2\Gurney180.jpg", (0, 0, 0), "Gurney"],
#                  ["Images2\Gurney270.jpg", (0, 0, 0), "Gurney"] ]

blurSize = (5,5)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-t", "--template", type=str, required=True,
#	help="path to input image where we'll apply template matching")
#ap.add_argument("-t", "--template", type=str, required=True,
#	help="path to template image")

ap.add_argument("-b", "--threshold", type=float, default=0.65,
	help="threshold for multi-template matching")
ap.add_argument("-v", "--visualize",
	help="Flag indicating whether or not to visualize each iteration")
args = vars(ap.parse_args())

# load the input image and template image from disk, then grab the
# template image spatial dimensions
print("[INFO] loading images...")

templates = []
templateBlur = []
templateSize = []
templateColor = []
numOfTemplate = len(templateInfoList)
for i in range(numOfTemplate):
    templates.append(cv2.imread(templateInfoList[i][0]))
    templateSize.append(templates[i].shape[:2])
    templateBlur.append(cv2.blur(templates[i], blurSize))
    templateColor.append(templateInfoList[i][2])
    
    
def templateMatch(img):
    color = ""
    result = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    for i in range(numOfTemplate):
        img = cv2.resize(img, (templates[i].shape[1], templates[i].shape[0]))
        template = cv2.cvtColor(templates[i], cv2.COLOR_BGR2GRAY)
        result.append([cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED), templateColor[i]])
    
    result.sort(key = lambda x : x[0], reverse = True)
    if result[0][0] >= args["threshold"]:
        color = result[0][1]
    else:
        color = "0" 
       
        
    return result[0][0], color

def ProcessGrid(image):
    resultList = []
    imagecopy = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

    # sort board
    board_rows = []
    row = []
    for (i, c) in enumerate(cnts, 1):
        area = cv2.contourArea(c+1)
        if area < 4000:
            row.append(c)
            (cnts, _) = contours.sort_contours(row, method="right-to-left")
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
            cv2.imshow("r",result)
            match, color = templateMatch(result)
            resultList.append(color)
            cv2.waitKey(300)
      
    return resultList

def split(arr, size):
     arrs = []
     while len(arr) > size:
         pice = arr[:size]
         arrs.append(pice)
         arr   = arr[size:]
     arrs.append(arr)
     return arrs[::-1]


while True:
        
    frame, image = cam.read()
    result = ProcessGrid(image)
    sliced = slice(7, len(result), 1 )
    result = result[sliced]
    boardArray = split(result[::-1], 6)
    print(np.matrix(boardArray))
   
    cv2.imshow("image", image)
    # cv2.waitKey()
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
        break
    
    
    
    
    
def HistMatch(img):

    # get image histogram
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    imghist2 = getHistogram(img)
    threshold = []
    for i in range(numOfTemplate):
        temp = templates[i]
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)
        temphist2 = getHistogram(temp)
        color = templateColor[i]
        threshold.append([compareHistogram(imghist2, temphist2), color])
        threshold.sort(key = lambda x : x[0], reverse = True)
    if threshold[0][0] > 1:
        return threshold[0]
    
        

def getHistogram(img):
    i=0
    hist = [0,0]
    while i < 2:
        hist[i] = cv2.calcHist([img],[i],None,[256],[0,256])
        cv2.normalize(hist[i], hist[i], 0, 1, cv2.NORM_MINMAX)
        i = i+1
    return hist

def compareHistogram(a, b):
    i=0
    result = 0
    while i < 2:
        result += cv2.compareHist(a[i], b[i], cv2.HISTCMP_CORREL)
        i = i+1
    return result


    