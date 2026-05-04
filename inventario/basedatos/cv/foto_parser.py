import datetime

import cv2
import numpy as np
from ultralytics import YOLO
from django.core.files.uploadedfile import UploadedFile

MODEL = YOLO('./basedatos/cv/my_model-n.pt', task='segment')

def convertir(fotos:list[UploadedFile]):
    imagenes = []
    for foto in fotos:
        foto.seek(0)
        bytes = foto.read()
        print(len(bytes))
        uint8_arr = np.frombuffer(bytes, np.uint8)
        img = cv2.imdecode(uint8_arr, cv2.IMREAD_COLOR)
        foto.seek(0)
        imagenes.append(img)
    return imagenes

def procesar(imagenes:list[cv2.typing.MatLike]):
    print('imgs: ', len(imagenes))
    i = 0
    for imagen in imagenes:
        results = MODEL(imagen, verbose=False)
        results[0].save('./output/'+ datetime.datetime.now().strftime("%d-%m-%Y_%H%M%S")+"_"+str(i)+".png")
        i+=1
