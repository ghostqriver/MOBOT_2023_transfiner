import skimage.io as io
import matplotlib.pyplot as plt
import cv2
from pycocotools.coco import COCO
import os
import random
import copy
import json
import tqdm
from shapely.geometry import Polygon

import detectron2
from detectron2.data import MetadataCatalog, DatasetCatalog
from .annos import get_coco_json_format

def coco_json_show(json_file,image_path,image_name=None):
    
    '''
    Given the path of json_file and images' path, random show 5 images with its annotations in the coco json file. If given a image_name, then only show that image.
    json_file: the path of the json
    image_path: the path contain the images in the json
    image_name: a certain file name, if given will only show this image

    '''

    coco = COCO(json_file)

    if image_name == None:
        image_ids = random.sample(coco.getImgIds(),5)
    else:
        image_ids = None
        for i in coco.getImgIds():
            if coco.loadImgs(i)[0]['file_name'] == image_name:
                image_ids = [i]
        if image_ids == None:
            raise ValueError('There is no '+image_name+' in dataset')
            
    for img in image_ids:

        imginfo = coco.loadImgs(img)[0]
        im = cv2.imread(os.path.join(image_path,coco.loadImgs(img)[0]['file_name']))
        annIds = coco.getAnnIds(img,catIds=[0,1])
        annsInfo = coco.loadAnns(annIds)
        img_name = coco.loadImgs(img)[0]['file_name']
    

        # Show the image
        plt.figure(figsize=(15,15))
        plt.imshow(im[:,:,[2,1,0]])
        plt.axis('off')
        coco.showAnns(annsInfo, True)

        # Show the text for each bbox
        coordinates=[]
        for j in range(len(annsInfo)):
            left = annsInfo[j]['bbox'][0]
            top = annsInfo[j]['bbox'][1]
            plt.text(left,top+15,coco.loadCats(annsInfo[j]['category_id'])[0]['supercategory'],fontsize=10)


        plt.title(img_name)
        plt.show()


def coco_json_read(json_file):
    
    '''
    Given the path of json_file, read the informations of this json
    json_file: the path of the json
    '''
    if type(json_file) is str:
        coco = COCO(json_file)
        print("In json file",json_file)
    else:
        coco = json_file
        

    print("*"*40)

    print("Images:",len(coco.getImgIds()))
    print("Annotations",len(coco.getAnnIds()))

    cat_ids = coco.getCatIds()
    print("Categories:",len(cat_ids))

    for i in cat_ids:
        print("\tCategory",i,":",len(coco.getAnnIds(catIds=[i])))
    
    print("*"*40)


def rm_cat_coco(json_path,cat_id):
    coco = COCO(json_path)
    print('In original json:')
    coco_json_read(coco) 

    json_ = json.load(open(json_path,'r'))
    coco_format = get_coco_json_format()
    coco_format['info'] = json_['info']
    coco_format['licenses'] = json_['licenses']
    coco_format['images'] = json_['images']
    cats = json_['categories']
    coco_format['categories'] = list(filter(lambda cat:cat['id'] != cat_id,cats))

    rm_cat = list(filter(lambda cat:cat['id'] == cat_id,cats))
    if len(rm_cat) != 1:
        raise BaseException('Does not exist this category in json')
    else:
        rm_cat = rm_cat[0]['name']
    new_json = json_path.split('.')[-2]+'_remove'+rm_cat

    annIds = coco.getAnnIds()
    coco_format['annotations'] = list(filter(lambda ann:ann['category_id'] != cat_id,[coco.loadAnns(annId)[0] for annId in annIds]))

    with open(new_json,"w") as outfile:
        json.dump(coco_format, outfile)
    print('Saved removed categroy',cat_id,'json file in',new_json)
    print('In new json:')
    coco_json_read(new_json) 

