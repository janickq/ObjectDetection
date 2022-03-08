import cv2
from imutils import contours
import numpy as np
from operator import itemgetter


# Load image, grayscale, and adaptive threshold
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

src = cv2.imread("Images2\image8.jpg")
src = cv2.resize(src, (640, 480))
# # Pi 
# templates = [ ["Images2/Red.jpg",  "Red"],
#                  ["Images2/Blue.jpg", "Blue"],
#                  ["Images2/Yellow.jpg", "Yellow"],
#                  ["Images2/Gurney.jpg", "Gurney"]]

                #  windows
templates = [ ["Images2\Red.jpg",  "Red"],
                 ["Images2\Blue.jpg", "Blue"],
                 ["Images2\Yellow.jpg", "Yellow"],
                 ["Images2\Gurney.jpg", "Gurney"]]

blurSize = (5,5)
numOfTemplate = len(templates)
    
def template(img):
    # get image histogram
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    threshold = []
    for i in range(numOfTemplate):
        
        temp = cv2.imread(templates[i][0])
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)
        temp = cv2.blur(temp, blurSize)
        color = templates[i][1]
        # w, h = temp.size
        img = cv2.resize(img, (temp.shape[0], temp.shape[1]))
        # get template histogram
        imghist2 = getHistogram(cv2.blur(img, blurSize))
        temphist2 = getHistogram(temp)
        # compare histogram
        threshold.append([compareHistogram(imghist2, temphist2), color])
        threshold.sort(key = lambda x : x[0], reverse = True) #true for correlation and intersect, false for the rest
        
    return threshold[0]
    
        

def getHistogram(img):
    i=0
    hist = [0,0]
    while i < 2: 
        hist[i] = cv2.calcHist([img],[i],None,[256],[0,256])
        cv2.normalize(hist[i], hist[i], alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX)
        i = i+1
    return hist

# change compare method here
def compareHistogram(a, b):
    i=0
    result = 0
    while i < 2:
        result += cv2.compareHist(a[i], b[i], cv2.HISTCMP_CORREL)
        i = i+1
    return result


src = cv2.cvtColor(cv2.imread("Images2\RedNew.png"), cv2.COLOR_BGR2HSV)
src2 = cv2.cvtColor(cv2.imread("Images2\Red.jpg"), cv2.COLOR_BGR2HSV)
src3 = cv2.cvtColor(cv2.imread("Images2\Blue.jpg"), cv2.COLOR_BGR2HSV)
while True:
    # i = 1000
    # print(template(cv2.imread("Images2\Yellow.jpg")))
    # cv2.waitKey(i)
    # print(template(cv2.imread("Images2\RedNew.png")))
    # cv2.waitKey(i)
    # print(template(cv2.imread("Images2\BlueNew.png")))
    # cv2.waitKey(i)
    # print(template(cv2.imread("Images2\GurneyNew.jpg")))
    # cv2.waitKey(i)
    # cv2.imshow("asd", src)
    # print(b,g,r)
    print(np.average(src))
    print(np.average(src2 ))
    print(np.average(src3))
    cv2.waitKey(1000)
    