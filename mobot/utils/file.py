'''
Functions for dealing with the images and masks, e.g. renaming to the same format, moving to the same folder.
It needed to be Customized it before applying.
'''


import glob
import os
import shutil

class file_process():
    def __init__(self):
        pass
    
    def read_folder(self,path=None):
        return os.listdir(path)
    
    def folder_rename(self,anno_folders):
        for anno_folder in anno_folders:
            new_name = anno_folder.split('.')[0][:-2] + (str(int(anno_folder.split('.')[0][-2:])))
            os.rename(anno_folder,new_name)
#             print(anno_folder,'-->'.new_name)
    
    def images_rename(self):
        for ims in glob.glob('2022-09-16/'+im_folder+'/images/*.png'):
            new_name = '2022-09-16/'+im_folder+'/images/'+im_folder+'_'+os.path.basename(ims)
            os.rename(ims,new_name)
        
    def move(self,anno_folders,im_folders):
        for anno_folder in anno_folders:
            for im_folder in im_folders:
                if anno_folder == im_folder:
#                     if os.path.exists(anno_folder+'/masks') and os.path.exists('2022-09-16/'+im_folder):
#                         shutil.move(anno_folder+'/masks','2022-09-16/'+im_folder+'/')
#                         print(anno_folder+'/masks --> '+ '2022-09-16/'+im_folder+'/')
#                     if os.path.exists(anno_folder+'/overlayed_masks') and os.path.exists('2022-09-16/'+im_folder):
#                         shutil.move(anno_folder+'/overlayed_masks','2022-09-16/'+im_folder+'/')
#                         print(anno_folder+'/overlayed_masks --> '+'2022-09-16/'+im_folder+'/')
                    pass

    def folder_remove(self,):
        for im_folder in os.listdir('2022-09-16'):
            if os.path.isdir('2022-09-16/'+im_folder):
                if os.path.exists('2022-09-16/'+im_folder+'/video'):
                    shutil.rmtree('2022-09-16/'+im_folder+'/video')
                    