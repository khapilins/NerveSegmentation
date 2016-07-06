from __future__  import division
import itertools
import random
from itertools import tee
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

def load_patient_mask_generator(patients_ids_list=[1],dropna=False, resize=(580, 420)):
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
    while True:
        for patient_id in patients_ids_list:         
            image=np.array([np.array(Image.open(('train/'+str(patient_id)+'_'+str(img)+'.tif')).resize(resize))[np.newaxis,:,:]/255\
                            for img in dat[dat.subject==patient_id].img])
            mask=np.array([np.array(Image.open('train/'+str(patient_id)+'_'+str(img)+'_mask.tif').resize(resize))[np.newaxis,:,:]/255\
                             for img in dat[dat.subject==patient_id].img])
            yield (image,mask)
            
def load_random_patient_mask_generator(patients_ids_list=[1],dropna=False, resize=(580, 420)):
    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    generator to load random image for subject from subject list
    input: patients_ids_list - list of integers - specifies subjects to load photos
           dropna - bool - specifies wether photos with empty masks should be dropped
           resize - tuple of int - desired size of output Image

    return: tuple of ndarrays, where first array is an array of photos of subject,
                second - its masks
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    dat=pd.read_csv('train_masks.csv')
    if dropna:
        dat=dat.dropna()
    while True:
        pat=random.choice(patients_ids_list)
        im=random.choice(list(dat[dat['subject']==pat].img))
        image=np.array(np.array(Image.open(('train/'+str(pat)+'_'+str(im)+'.tif')).resize(resize))[np.newaxis,:,:]/255)
        mask=np.array(np.array(Image.open('train/'+str(pat)+'_'+str(im)+'_mask.tif').resize(resize))[np.newaxis,:,:]/255)
        yield (image,mask)   
    
def run_length_to_image(pixels_string, shape=(580, 420)):
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

def image_to_run_lenght(im_array, shape=(1,580, 420)):
    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    function to convert image mask to run length encoded string
    input: im_array - numpy.ndarray
           shape - image shape tuple to resize if im_array is smaller then actual image           

    returns string
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    if im_array.shape!=(shape[1],shape[0]):
        image=Image.fromarray(im_array[0])
        image=image.resize((shape[1],shape[2]))
        im_array=np.array(image)
    
    im_array=im_array.flatten(order='F')
    start_run_length=0    
    run_length=0
    indexes=[]    
    for index,value in enumerate(im_array):                
        if value!=0:            
            if start_run_length==0:
                start_run_length=index+1
            run_length+=1
        else:
            if start_run_length!=0:
                indexes.append((start_run_length,run_length))
                start_run_length=0
                run_length=0
    
    run_length_string=' '.join(map(str,itertools.chain.from_iterable(indexes)))
    return run_length_string

def train_test_generator(train_size=0.8,dropna=False, resize=(580, 420),shuffle=True,random_seed=None):
    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    function to make generators for train and test data
    input: 0<train_size<1 - float - percentaage of train data
           dropna - bool - specifies wether photos with empty masks should be dropped
           resize - tuple of int - desired size of output Image
           shuffle - bool - wether it's needed to shuffle data  
           random_seed - int or None - sets seed of shuffling, if None system time used

    returns tuple of generators
    first element - generator of train images
    first element - generator of test images
    generators are made with help of load_patient_mask_generator()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    dat=pd.read_csv('train_masks.csv')
    subjects=dat.subject.unique()
    if shuffle:
        random.seed(random_seed)
        random.shuffle(subjects)
    train_subjects=subjects[:int(round(train_size*len(subjects)))]
    test_subjects=subjects[int(round(train_size*len(subjects))):]
    train_generator=load_patient_mask_generator(train_subjects,dropna=dropna,resize=resize)
    test_generator=load_patient_mask_generator(test_subjects,dropna=dropna,resize=resize)
    return train_generator,test_generator

def random_photo_train_test_gen(train_size=0.8,dropna=False, resize=(580, 420),shuffle=True,random_seed=None):
    dat=pd.read_csv('train_masks.csv')
    subjects=dat.subject.unique()
    if shuffle:
        random.seed(random_seed)
        random.shuffle(subjects)
    train_subjects=subjects[:int(round(train_size*len(subjects)))]
    test_subjects=subjects[int(round(train_size*len(subjects))):]
    train_generator=load_random_patient_mask_generator(train_subjects,dropna=dropna,resize=resize)
    test_generator=load_random_patient_mask_generator(test_subjects,dropna=dropna,resize=resize)
    return train_generator,test_generator
    
if __name__=='__main__':
    _,im=next(load_patient_mask_generator([1]))    
    image_to_run_lenght(im[0])
    a=load_random_patient_mask_generator()
    print next(a)
    #train,test=train_test_generator()
    #print next(train)
    #print next(test)