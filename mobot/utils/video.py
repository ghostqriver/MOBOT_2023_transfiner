import glob
import cv2 as cv
import tqdm
import os 


def images_rename(path):
    '''

    Renaming images in several folders
    
    '''
    for ind,ims in enumerate(glob.glob(path+'/*.png')):
        new_name = os.path.join(path,str(ind)+'.png')
        os.rename(ims,new_name)


def frame2videos(path=None):
    '''

    transform multiple folder's images into videos

    '''
    for i in os.listdir(path):
        images = list(glob.glob(i+'/*.png'))
        images.sort(key=lambda a:int(a.split('\\')[1].split('.')[0])) # don't forget sort frames, or 10 will before 2
        frame2video(images)


def frame2video(frame_path,file_name=None):
    '''
    
    frame_path: a folder path contain png images or a list of images path
    
    ''' 
    if type(frame_path) == str:
        frames = glob.glob(frame_path+'/*.png')
        file_name = frame_path+os.path.dirname(frame_path)+'.mp4'
    else:
        
        if len(frame_path) == 0:
            return
        frames = frame_path
        file_name = os.path.join(os.path.dirname(frame_path[0]),os.path.dirname(frame_path[0])+'.mp4')
        
    
    img0 = cv.imread(frames[0])

    height,width = img0.shape[:2]

    video_writer = cv.VideoWriter(file_name, fourcc=cv.VideoWriter_fourcc(*"mp4v"), fps=5, frameSize=(width, height), isColor=True)

    for frame in tqdm.tqdm(frames):
#         print(frame)
        img = cv.imread(frame)

        video_writer.write(img)

    video_writer.release()

    print('Saved video as',file_name)

    return file_name


def read_video_info(video_path):
        '''

        Read the video's frame, size, FPS
        
        '''
        cap = cv.VideoCapture(video_path)
        frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
#         width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
#         height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
#         fps = cap.get(cv.CAP_PROP_FPS)
#         duration = frames/fps
#         print('In the video',video_path+':')
        print('Frames:',int(frames))
#         print('Size:',int(width),'*',int(height))
#         print('FPS:',fps)
#         print('Duration:',duration)
#         return frames,(width,height),fps,duration