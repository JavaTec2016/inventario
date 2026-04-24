import cv2

from inventario.cv.train import TRAINED_NAME, getDatasetLabels, instanciar


def draw_boxes(image, boxes, labels):
    image_h, image_w, _ = image.shape

    for box in boxes:
        xmin = int(box.xmin*image_w)
        ymin = int(box.ymin*image_h)
        xmax = int(box.xmax*image_w)
        ymax = int(box.ymax*image_h)

        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,255,0), 3)
        cv2.putText(image, 
                    labels[box.get_label()] + ' ' + str(box.get_score()), 
                    (xmin, ymin - 13), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1e-3 * image_h, 
                    (0,255,0), 2)
        
    return image

yolo = instanciar()
yolo.load_weights(TRAINED_NAME)
def detect(img_path=''):
    image = cv2.imread(img_path)
    boxes = yolo.predict(image)
    image = draw_boxes(image, boxes, getDatasetLabels())
    print('Detecciones: ', len(boxes))
    #lo saca como imagen
    cv2.imwrite(img_path[:-4] + '_detected' + img_path[-4:], image)