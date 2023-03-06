'''
Functions to transform the mask annotations to coco-json.
Source: https://github.com/chrise96/image-to-coco-json-converter
Customize the global parameters before call the functions.
E.g.
coco_transformer('')

'''

category_ids = {
       #"background":0,
        "end":0,
        "side":1
}

# Define which colors match which categories in masks
category_colors = {
     "(128, 0, 0)": 0, # end
     #"(0, 128, 0)": 1, # side
}

# Background use this to ignore the color in the mask which we will not consider it as a category
# Don't forget the space before the ,
ignore_colors = "(0, 0, 0)"

# Define the ids that are a multiplolygon. 
# Stuff: will not be seperated, In our case: background, all color will be seem as one， but here we ignore background
multipolygon_ids = [] 


from PIL import Image                                      # (pip install Pillow)
import numpy as np                                         # (pip install numpy)
from skimage import measure                                # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon         # (pip install Shapely)
import os
import json

import glob
from tqdm import tqdm

def create_sub_masks(mask_image, width, height):
    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))[:3]

            # Check to see if we have created a sub-mask...
            pixel_str = str(pixel)
            sub_mask = sub_masks.get(pixel_str)
            if sub_mask is None:
               # Create a sub-mask (one bit per pixel) and add to the dictionary
                # Note: we add 1 pixel of padding in each direction
                # because the contours module doesn"t handle cases
                # where pixels bleed to the edge of the image
                sub_masks[pixel_str] = Image.new("1", (width+2, height+2))

            # Set the pixel value to 1 (default is 0), accounting for padding
            sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks

def create_sub_mask_annotation(sub_mask):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(np.array(sub_mask), 0.5, positive_orientation="low")
    # print(contours)
    polygons = []
    segmentations = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        # print('hello')
        # print(contour)
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        
        if(poly.is_empty):
            # Go to next iteration, dont save empty values in list
            continue
        # print(poly)
        polygons.append(poly)
        
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)
    
    return polygons, segmentations

def create_category_annotation(category_dict):
    category_list = []

    for key, value in category_dict.items():
        category = {
            "supercategory": key,
            "id": value,
            "name": key
        }
        category_list.append(category)

    return category_list

def create_image_annotation(file_name, width, height, image_id):
    images = {
        "file_name": file_name,
        "height": height,
        "width": width,
        "id": image_id
    }

    return images

def create_annotation_format(polygon, segmentation, image_id, category_id, annotation_id):
    min_x, min_y, max_x, max_y = polygon.bounds
    width = max_x - min_x
    height = max_y - min_y
    bbox = (min_x, min_y, width, height)
    area = polygon.area

    annotation = {
        "segmentation": segmentation,
        "area": area,
        "iscrowd": 0,
        "image_id": image_id,
        "bbox": bbox,
        "category_id": category_id,
        "id": annotation_id
    }

    return annotation

def get_coco_json_format():
    # Standard COCO format 
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}]
    }

    return coco_format
    
    
def images_annotations_info(maskpath):
    # Get "images" and "annotations" info 
    # This id will be automatically increased as we go
    annotation_id = 0
    image_id = 0
    annotations = []
    images = []
    
    log_file = maskpath + 'log.txt'
    
    for mask_image in tqdm(glob.glob(maskpath + "*.png")):
        
        try:
            # The mask image is *.png but the original image is *.jpg.
            # We make a reference to the original file in the COCO JSON file

            # The original_file_name should be same to the maskname
            original_file_name = os.path.basename(mask_image)#.split(".")[0] + ".jpg"

            # Open the image and (to be sure) we convert it to RGB
            mask_image_open = Image.open(mask_image).convert("RGB")
            w, h = mask_image_open.size

            # "images" info 
            image = create_image_annotation(original_file_name, w, h, image_id)
            images.append(image)

            sub_masks = create_sub_masks(mask_image_open, w, h)

            # Remove the ignore_colors in the sub_masks then it will not be generated in the final json file
            if ignore_colors in sub_masks.keys():
                sub_masks.pop(ignore_colors)

            for color, sub_mask in sub_masks.items(): 
                category_id = category_colors[color]

                # "annotations" info
                polygons, segmentations = create_sub_mask_annotation(sub_mask)

                # Check if we have classes that are a multipolygon
                if category_id in multipolygon_ids:
                    # Combine the polygons to calculate the bounding box and area
                    multi_poly = MultiPolygon(polygons)

                    annotation = create_annotation_format(multi_poly, segmentations, image_id, category_id, annotation_id)

                    annotations.append(annotation)
                    annotation_id += 1
                else:
                    for i in range(len(polygons)):
                        # Cleaner to recalculate this variable
                        segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]

                        annotation = create_annotation_format(polygons[i], segmentation, image_id, category_id, annotation_id)

                        annotations.append(annotation)
                        annotation_id += 1
            image_id += 1
        
        # Record a log which shows whether an mask transform to json successfully or not, if not record the error info
        # in case it stops when meet the error then lose all json content which already transformed
        # to keep the json to be usable, we may delete those error images in the final dataset
        except  BaseException as e:
            with open(log_file,'a') as log:
                log.write(mask_image+': '+str(e)+'\n')
            continue
            
        else:
            with open(log_file,'a') as log:
                log.write(mask_image+': successfully\n')
                
    return images, annotations, annotation_id
    
    
def coco_transformer(mask_path,json_file=None):
    
    if json_file_name is None:
        json_file = mask_path + mask_path.split('/')[-2] + '.json'
    
    #Create annotation
    coco_format = get_coco_json_format()

    # Create category section
    coco_format["categories"] = create_category_annotation(category_ids)

    # Create images and annotations sections
    coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(mask_path)

    with open(json_file,"w") as outfile:
        json.dump(coco_format, outfile)

    print("Created %d annotations for images in folder: %s, saved in %s"% (annotation_cnt, mask_path,json_file))