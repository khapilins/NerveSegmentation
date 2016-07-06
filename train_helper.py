from __future__ import division
import loader
import numpy as np

from keras.layers import Convolution2D,MaxPooling2D,UpSampling2D,Flatten,Dense
from keras.datasets import mnist
import keras.backend as K
import keras.callbacks as C
from keras.models import Sequential
from keras.layers import Activation, Dropout
from keras.layers.advanced_activations import ELU

def get_model(shape=(1,420,580)):    
    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Function for model building
    Also prints output shapeafter each layer
    For prototyping model structure for now
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
    model=Sequential()
    model.add(Convolution2D(32,3,3, input_shape=shape,border_mode='same'))
    model.add(Activation(ELU()))
    print model.layers[-1].output_shape
    model.add(MaxPooling2D((2,2)))
    print model.layers[-1].output_shape
    
    model.add(Convolution2D(32,3,3,border_mode='same'))
    model.add(Activation(ELU()))
    print model.layers[-1].output_shape
    model.add(MaxPooling2D((2,2)))
    print model.layers[-1].output_shape
    
    model.add(UpSampling2D((2,2)))
    print model.layers[-1].output_shape
    model.add(Convolution2D(32,3,3,border_mode='same'))
    model.add(Activation(ELU()))
    print model.layers[-1].output_shape
    model.add(UpSampling2D((2,2)))
    print model.layers[-1].output_shape
    model.add(Convolution2D(32,3,3,border_mode='same'))
    model.add(Activation(ELU()))
    print model.layers[-1].output_shape    
    model.add(Convolution2D(1,1,1))
    model.add(Activation('sigmoid'))
    print model.layers[-1].output_shape
    
    return model

def dice_loss_K(y_true,y_pred):

    y_true=K.flatten(y_true)
    y_pred=K.flatten(y_pred)
    intersection=K.sum(y_true*y_pred)
    denominator=(K.sum(y_true)+K.sum(y_pred))
    score=(2*intersection)/denominator
    return 1-score

def linear_scheduler(epoch,M=1000):
    return model.optimizer.get_value()*(1-epoch/M)
    

def train_model(model, train_gen,\
                val_gen,\
                optimizer='adadelta',\
                nb_epochs=70,\
                early_stopping_patience=4,\
                model_name='model',\
                scheduler=linear_scheduler,
                loss=dice_loss_K):
    """Helper function for training"""

    try:
        model.compile(loss=loss, optimizer=optimizer)
        
        if scheduler:
            change_lr = LearningRateScheduler(scheduler)
        else:
            change_lr=LearningRateScheduler(lambda x:model.optimizer.lr.get_value())
        
        remote = callbacks.RemoteMonitor(root='http://localhost:9000')
        check=callbacks.ModelCheckpoint(model_name+'checkpoint', monitor=loss, verbose=0, save_best_only=True, mode='min')
    
    
        history=model.fit_generator(train_gen,
                        samples_per_epoch=1000,nb_epoch=nb_epochs,
                  callbacks=[check,remote,keras.callbacks.EarlyStopping(monitor='val_loss',\
                  patience=early_stopping_patience, mode='auto'),change_lr],\
                  validation_data=val_gen,nb_val_samples=2000,metrics=[loss])
    except KeyboardInterrupt:
        print
        print '---------------------------'
        print 'Training terminated by user'
        model.save_weights(model_name+'.h5') 