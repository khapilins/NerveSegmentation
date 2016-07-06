from loader import *
from train_helper import *

def tmp_gen(gen):
    while 1:
        a=next(gen)
        yield (a[0][np.newaxis,:,:],a[1][np.newaxis,:,:])

if __name__=='__main__':
    train_gen,test_gen=loader.random_photo_train_test_gen(resize=(140, 100),random_seed=42)     
    model = get_model(shape=(1,100,140))
    train_model(model,tmp_gen(train_gen),tmp_gen(test_gen))
