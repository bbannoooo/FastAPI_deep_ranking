import numpy as np
import pandas as pd
from typing import List
from keras.applications.vgg16 import VGG16
from keras.layers import *
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from skimage import transform
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Embedding
from keras import backend as K
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from io import BytesIO
from PIL import Image
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

def convnet_model_():
    vgg_model = VGG16(weights=None, include_top=False)
    x = vgg_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Lambda(lambda  x_: K.l2_normalize(x,axis=1))(x)
    convnet_model = Model(inputs=vgg_model.input, outputs=x)
    return convnet_model

def deep_rank_model():
  
  convnet_model = convnet_model_()
  first_input = Input(shape=(256,256,3))
  first_conv = Conv2D(96, kernel_size=(8, 8),strides=(16,16), padding='same')(first_input)
  first_max = MaxPool2D(pool_size=(3,3),strides = (4,4),padding='same')(first_conv)
  first_max = Flatten()(first_max)
  first_max = Lambda(lambda  x: K.l2_normalize(x,axis=1))(first_max)

  second_input = Input(shape=(256,256,3))
  second_conv = Conv2D(96, kernel_size=(8, 8),strides=(32,32), padding='same')(second_input)
  second_max = MaxPool2D(pool_size=(7,7),strides = (2,2),padding='same')(second_conv)
  second_max = Flatten()(second_max)
  second_max = Lambda(lambda  x: K.l2_normalize(x,axis=1))(second_max)

  merge_one = concatenate([first_max, second_max])

  merge_two = concatenate([merge_one, convnet_model.output])
  emb = Dense(4096)(merge_two)
  l2_norm_final = Lambda(lambda  x: K.l2_normalize(x,axis=1))(emb)

  final_model = Model(inputs=[first_input, second_input, convnet_model.input], outputs=l2_norm_final)

  return final_model

model = deep_rank_model()
weights = 'deepranking_test_early.h5'

model.load_weights(weights)

app = FastAPI()

@app.post("/similarity/")
async def create_upload_files(files: List[UploadFile] = File(...), input_file: UploadFile = File()):
  img = await input_file.read()
  img = Image.open(BytesIO(img))
  img = img_to_array(img).astype("float64")
  img = transform.resize(img, (256, 256))
  img *= 1. / 255
  img = np.expand_dims(img, axis = 0)

  embedding1 = model.predict([img, img, img])[0]

  distance_t= []
  filename = []
  for img2 in files:
    img3 = await img2.read()
    img3 = Image.open(BytesIO(img3))
    img3 = img_to_array(img3).astype("float64")
    img3 = transform.resize(img3, (256, 256))
    img3 *= 1. / 255
    img3 = np.expand_dims(img3, axis = 0)

    embedding2 = model.predict([img3, img3, img3])[0]

    distance = sum([(embedding1[idx] - embedding2[idx])**2 for idx in range(len(embedding1))])**(0.5)
    distance_t.append(distance)
    filename.append(img2.filename)
    print(img2.filename)
  metadata = pd.DataFrame({"distance": distance_t, "filename": filename})

  metadata.sort_values("distance", inplace=True)
  metadata.reset_index(drop=True)
  # print('Euclidean Distance: {0}, filename: {1}'.format(metadata['distance'].values, metadata['filename'].values))
  print(metadata['filename'])
  return {'rank1': metadata['filename'].values[1], 'rank2': metadata['filename'].values[2], 'rank3': metadata['filename'].values[3]}

@app.get("/")
async def main():
    content = """
<body>
<form action="/similarity/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input name="input_file" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)





