from tqdm import tqdm 
import tensorflow as tf 
from keras.applications.nasnet import NASNetLarge 
from keras.layers import Flatten, Input
from keras.models import Model
from keras.preprocessing import image
from keras.applications.nasnet import preprocess_input
import numpy as np
#from google.cloud import storage 
from io import BytesIO
import time
import os
import pandas as pd
start = time.time()
import keras
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

if __name__ == "__main__":

  model = NASNetLarge (weights='imagenet', pooling=max, include_top = True) 
  ex_folders=[]
  iter=0
  df=pd.DataFrame()
  path='/content/drive/MyDrive/folders'
  folders=os.listdir(path)
  config = tf.compat.v1.ConfigProto(device_count = {'GPU': 1 , 'CPU': 56} ) 
  sess = tf.compat.v1.Session(config=config) 
  keras.backend.set_session(sess) 

  for i in tqdm(range(len(folders))):
        path='/content/drive/MyDrive/folders/'+str(i+1)+'/news_img.jpg' 
        img = Image.open(path)
        img = img.resize((331,331), Image.ANTIALIAS)
        img = img.convert('RGB')
        x = image.img_to_array(img)
        ##print('here')
        x = np.expand_dims(x, axis=0) 
        x = preprocess_input(x) 
        features = model.predict(x) 
        features_reduce = features.squeeze()
        df[i]=features_reduce
        if(i%100==0):
          print(i)

  filename="visual_features_resnet.csv"
  df.to_csv(filename,index=False)  
  
