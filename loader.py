from __future__  import division
from itertools import tee
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

def load_patient_mask_generator(patients_ids_list=[1],dropna=False, resize=(420, 580)):
    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    generator to load all images and masks of a single subject from list of subjects
    input: patients_ids_list - list of integers - specifies subjects to load photos
           dropna - bool - specifies wether photos with empty masks should be dropped
           resize - tuple of int - desired size of output Image

    return: tuple of ndarrays, where first array is an array of photos of subject,
                second - its masks
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~""" 
    dat=pd.read_csv('train_masks.csv')
    if dropna:
        dat=dat.dropna()
    for patient_id in patients_ids_list:         
        image=np.array([np.array(Image.open(('train/'+str(patient_id)+'_'+str(img)+'.tif')).resize(resize))\
                        for img in dat[dat.subject==patient_id].img])
        mask=np.array([np.array(Image.open('train/'+str(patient_id)+'_'+str(img)+'_mask.tif').resize(resize))\
                         for img in dat[dat.subject==patient_id].img])
        yield (image,mask)
    
def mask_to_image(pixels_string, shape=(420, 580)):
    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    function to convert pixels string from train_mask.csv to numpy array
    input: pixels_string - string
           shape - image shape tuple

    returns numpy.ndarray
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    x=map(int,pixels_string.split())    
    x1=x[::2]
    del x[::2]
    x2=x        
    pix=zip(x1,x2)    
    im=np.zeros(shape)
    im=im.flatten('F')
    for i in pix:
        im[i[0]-1:i[0]+i[1]]=1
    im=im.reshape(shape,order='F')
    return im


if __name__=='__main__':
    for i in load_patient_mask_generator([4,5,10]):
        a,b=i
        print a[0]
        print b[0]