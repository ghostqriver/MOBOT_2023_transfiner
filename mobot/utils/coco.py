import skimage.io as io
import matplotlib.pyplot as plt
import cv2
from pycocotools.coco import COCO
import os
import random
import copy
import json
import tqdm

import detectron2
from detectron2.data import MetadataCatalog, DatasetCatalog


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

    coco = COCO(json_file)

    print("In json file",json_file)
    print("*"*40)

    print("Images:",len(coco.getImgIds()))
    print("Annotations",len(coco.getAnnIds()))

    cat_ids = coco.getCatIds()
    print("Categories:",len(cat_ids))

    for i in cat_ids:
        print("\tCategory",i,":",len(coco.getAnnIds(catIds=[i])))
    
    print("*"*40)


def coco_rm_cat(json_file,image_root):
    "A template for removing specific annotations in the coco json file"
    json_ = detectron2.data.datasets.load_coco_json(json_file, image_root, "MOBOT_Train", extra_annotation_keys=None)
    new_json = []
    print('Filtering...')
    for d in tqdm.tqdm(json_):
        d_ = copy.deepcopy(d)
        new_anno = []
        num_side = 0
        for ind,anno in enumerate(d['annotations']):
            class_ = anno['category_id']
            if class_ == 1: 
                num_side += 1
                pass
            else:
                anno['image_id'] = d['image_id'] 
                new_anno.append(anno)
        d['annotations'] = new_anno
        new_json.append(d)
        
    new_annotations = []
    id = 0
    print('Transforming...')
    for j in tqdm.tqdm(new_json):
        for anno in j['annotations']:
            anno_ = copy.deepcopy(anno)
            anno_['id'] = id
            del anno_['bbox_mode']
            new_annotations.append(anno_)
            id += 1

    print('Saving...')
    with open(json_file,'r+') as file:
        content=file.read()

    json_origin=json.loads(content) # modify the json_origin['annotations']

    json_origin['annotations'] = new_annotations

    new_json=json.dumps(json_origin)

    with open(json_file.split('.')[0]+'_end.json','w+') as file:
        file.write(new_json)

    print('Saved in',json_file.split('.')[0]+'_end.json')