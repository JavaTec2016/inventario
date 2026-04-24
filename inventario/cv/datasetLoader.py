import os

import numpy as np

xml_name = 'si'   
img_name = 'si'
#de la BD
labels = [
    "cafe_oro"
]
xml_dir = 'annotation/' + xml_name
img_dir = 'annotation/' + img_name

size = 3120 #a
trained_name = 'cnn_inventario.h5'

def leer_annotations(xmls_path, imgs_path, labels=[]):
    all_imgs = []
    seen_labels = {}
    for annot in sorted(os.listdir(xmls_path)):
        img = {'object':[]}
        tree = ET.parse(xmls_path, annot)

        #==========lectura de atributos de la imagen
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = imgs_path + elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            #====aqui agarra los objetos anotados por separao
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1
                        
                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]

                    #==========agarra los bounds
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))
        
        if len(img['object']) > 0:
            all_imgs += [img]
    return all_imgs, seen_labels


#=======CARGAR DATASET

train_imgs, train_labels = leer_annotations(xml_dir, img_dir, labels)
print('imagenes', len(train_imgs), 'labels', len(train_labels))

#========separar entrenadera y validacion

train_valid_split = int(0.8*len(train_imgs))
np.random.shuffle(train_imgs)
valid_imgs = train_imgs[train_valid_split:]
train_imgs = train_imgs[:train_valid_split]
print('traineo: ', len(train_imgs), 'validacion: ', len(valid_imgs))