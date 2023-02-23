import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def load_data(images_dir):
    '''
    Loads images from path. Images should be structured as following
    00001_c.png, 00001_s.png, 00001_g.png.

    *_c denotes color images, *_s denotes saliency frames, *_g denotes ground truth frames. 
    '''
    images = [images_dir+"/"+f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    print(f"Total number of images: {len(images)//3}")

    images_saliency = np.zeros((len(images)//3},240,480*2))
    images_color = np.zeros((len(images)//3},240,480*2,3))

    for i in tqdm(range(0,len(images))):
        if i%3==0:
            images_color[i//3,:240,:480] = np.asarray(Image.open(images[i]))
            images_color[i//3,:240,480:480*2] = np.asarray(Image.open(images[i]))
        if i%3==1:
            images_saliency[i//3,:240,:480] = np.asarray(Image.open(images[i]).convert("L"))
            images_saliency[i//3,:240,480:480*2] = np.asarray(Image.open(images[i]).convert("L"))
    
    return images_color, images_saliency
        