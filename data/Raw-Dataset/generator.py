import numpy as np
import os
import random
from tqdm.auto import tqdm
import cv2
import pyarrow.parquet as pq
import glob
import math
import cupy as cp
from einops import rearrange

BATCH_SIZE = 2048
def generate(pf, path, a):
    record_batch = pf.iter_batches(batch_size=BATCH_SIZE)
    while True:
        try:

            batch = next(record_batch)
            a = transform(batch, path,a)

        except StopIteration:
            # print("Done")
            return a

def transform(batch,path, a):
    p = batch.to_pandas()
    im = cp.array(np.array(np.array(p.iloc[:, 3].tolist()).tolist()).tolist())
    meta = np.array(p.iloc[:,0])
    return saver(im,meta, path, a)


def saver(im, meta, path, a):
    alpha = a

    im[im < 1.e-3] = 0 #Zero_suppression
    im = rearrange(im, 'b c h w -> b h w c')
    for _z in range(13):
        im[:,:,:,_z] = (im[:,:,:,_z] - im[:,:,:,_z].mean())/(im[:,:,:,_z].std())
        im[:,:,:,_z] = cp.clip(im[:,:,:,_z], a_min = 0, a_max = 500*im[:,:,:,_z].std(axis = (1,2))[:,None,None])

    im = im.get()
    for i in range(meta.shape[0]):
        img = im[i,:,:,:]
        for _z in range(13):
            
            if (img[:,:,_z]==0).all() == False:
                img[:,:,_z] = 255*(img[:,:,_z])/(img[:,:,_z].max())



        img = img.astype(np.uint8)
        img = rearrange(img, 'h w c -> h ( w c )')
        alpha = alpha +1
        impath = os.path.join(path,str(str(alpha)+".png"))
        cv2.imwrite(impath , img)
    return alpha


def runner(source, target):
    """
    Fuction to convert all the Parquet Files in a given folder to .png format Files
    Args:
    source: The souce folder of the Parquet Files
    target: The target folder where the dataset will be stored
    """
    a = 0
    
    files = os.listdir(source)
    print("The following files were found in the provided Directory")
    print(files)
    for i in range(len(files)):
        a = generate(pq.ParquetFile(os.path.join(source, files[i])), target, a)
    print("The files were successfully generated")
    

if __name__ == "__main__":
    source = input("Enter the source:") 
    target = input("Enter the target:")
    runner(source,target)