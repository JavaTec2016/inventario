#=====feed en vivo (usar IP Webcam)
import requests
import cv2
import numpy
import imutils
url = 'http://192.168.1.6:8080/shot.jpg'

while True:
    response = requests.get(url)
    img_buffer = numpy.array(bytearray(response.content), dtype=numpy.uint8)
    frame = cv2.imdecode(img_buffer, -1)
    frame = imutils.resize(frame, width=1280, height=1280)

    cv2.imshow('camera', frame)

    if(cv2.waitKey(1) == 27):
        break

cv2.destroyAllWindows()