
import cv2
import numpy as np
# HSV Control using Trackbars
# read a colourful image
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# define a null callback function for Trackbar
def null(x):
    pass

# create six trackbars for H, S and V - lower and higher masking limits 
cv2.namedWindow('HSV')
# arguments: trackbar_name, window_name, default_value, max_value, callback_fn
cv2.createTrackbar("HL", "HSV", 0, 179, null)
cv2.createTrackbar("HH", "HSV", 179, 179, null)
cv2.createTrackbar("SL", "HSV", 0, 255, null)
cv2.createTrackbar("SH", "HSV", 255, 255, null)
cv2.createTrackbar("VL", "HSV", 0, 255, null)
cv2.createTrackbar("VH", "HSV", 255, 255, null)

while True:
    frame_got, img = cam.read()
    if frame_got is False:
        break

    img = cv2.blur(img, (3,3))
    # convert BGR image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # read the Trackbar positions
    hl = cv2.getTrackbarPos('HL','HSV')
    hh = cv2.getTrackbarPos('HH','HSV')
    sl = cv2.getTrackbarPos('SL','HSV')
    sh = cv2.getTrackbarPos('SH','HSV')
    vl = cv2.getTrackbarPos('VL','HSV')
    vh = cv2.getTrackbarPos('VH','HSV')

    # create a manually controlled mask
    # arguments: hsv_image, lower_trackbars, higher_trackbars
    mask = cv2.inRange(hsv, np.array([hl, sl, vl]), np.array([hh, sh, vh]))
    # derive masked image using bitwise_and method
    final = cv2.bitwise_and(img, img, mask=mask)

    # display image, mask and masked_image 
    cv2.imshow('Original', img)
    cv2.imshow('Mask', mask)
    cv2.imshow('Masked Image', final)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()