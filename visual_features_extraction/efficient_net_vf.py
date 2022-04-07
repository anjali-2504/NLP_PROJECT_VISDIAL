from tqdm import tqdm 
import tensorflow as tf 
from keras.applications.efficientnet import EfficientNetB7
from keras.layers import Flatten, Input
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
#from google.cloud import storage 
from io import BytesIO
import time
#import cv2
import pandas as pd
start = time.time()
model = EfficientNetB7(include_top=True, weights="imagenet",pooling=max)
ex_folders=[]
iter=0
df=pd.DataFrame()
import os
path='/home/puneet/code/auto_eval/feedback_module./data'
folders=os.listdir(path)
from PIL import Image

for i in range(len(folders)):
    #iter=iter+1
    #try:
        #print(iter)
        path='/home/puneet/code/auto_eval/feedback_module./data/'+str(i+1)+'/news_img.jpg' 
        img = Image.open(path)
        img = img.resize((600,600), Image.ANTIALIAS)
        img = img.convert('RGB')
        x = image.img_to_array(img)
        ##print('here')
        x = np.expand_dims(x, axis=0) 
        x = preprocess_input(x) 
        features = model.predict(x) 
        features_reduce = features.squeeze()
        print(features_reduce.shape)
        df[i]=features_reduce
filename="visual_features_efficientnet.csv"
df.to_csv(filename,index=False)
print(len(ex_folders)) 
