import datetime
import json

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

def contabilizar(results:list):
    resultConteos = {}
    for res in results:
        for clase in json.loads(res.to_json()):
            resultConteos[clase['name']] = resultConteos.get(clase['name'], 0)+1
    return resultConteos

def procesar(imagenes:list[cv2.typing.MatLike]):
    print('imgs: ', len(imagenes))
    i = 0
    results = MODEL(imagenes, verbose=True)
    print('Contando objetos...')
    resultConteos = contabilizar(results)
    for res in results:
        #res.save('./output/'+ datetime.datetime.now().strftime("%d-%m-%Y_%H%M%S")+"_"+str(i)+".png")
        i+=1
    print(resultConteos)
    return resultConteos
