from imutils.object_detection import non_max_suppression
import numpy as np
import datetime
import imutils
import urllib
import cv2

stream = urllib.urlopen('http://192.168.1.101/mjpg/1/video.mjpg')

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def extractPart(frame, x, y, w, h):
    return frame[y: y + h, x: x + w]

def resize(frame, width):
    return imutils.resize(frame, width=min(width, frame.shape[1]))

def process(frame):
    frame = extractPart(frame, 5, 400, 1900, 900)
    image = resize(frame, 700)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (rects, weights) = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.02)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    if (len(pick) > 0):
        pictureFile = 'imgs/{:%Y%m%d%H%M%S}.png'.format(datetime.datetime.now())
        print("[INFO] detected people: {}. Picture saved: {}".format(len(pick), pictureFile))
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        cv2.imwrite(pictureFile, image)

    cv2.imshow('cam', image)

bytes = ''
while True:
    bytes += stream.read(1024)
    a = bytes.find('\xff\xd8')
    b = bytes.find('\xff\xd9')
    if a != -1 and b != -1:
        jpg = bytes[a: b + 2]
        bytes = bytes[b + 2:]

        frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.CV_LOAD_IMAGE_COLOR)

        process(frame)
        if cv2.waitKey(1) == 27:
            exit(0)
