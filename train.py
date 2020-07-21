import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
import os
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
import random
from model.resnet50 import Resnet50

epochs = 50
batch_size = 32
batch_size_val = 16
path_image = "dataset/JPEGImages"
path_train = "dataset/ImageSets/Main/NNEW_trainval_3.txt"
path_val = "dataset/ImageSets/Main/NNEW_test_3.txt"

fp_train = open(path_train, 'r')
filenames_train = []
mild_train = []
moderate_train = []
severe_train = []
verysevere_train = []
lesions_train = []
for line in fp_train.readlines():
    filename_train, label_train, lesion_train = line.split()
    filenames_train.append(filename_train)
    if label_train == '0':
        mild_train.append(1)
        moderate_train.append(0)
        severe_train.append(0)
        verysevere_train.append(0)
    elif label_train == '1':
        mild_train.append(0)
        moderate_train.append(1)
        severe_train.append(0)
        verysevere_train.append(0)
    elif label_train == '2':
        mild_train.append(0)
        moderate_train.append(0)
        severe_train.append(1)
        verysevere_train.append(0)
    else:
        mild_train.append(0)
        moderate_train.append(0)
        severe_train.append(0)
        verysevere_train.append(1)
    lesions_train.append(lesion_train)
fp_train.close()

fp_val = open(path_val, 'r')
filenames_val = []
mild_val = []
moderate_val = []
severe_val = []
verysevere_val = []
lesions_val = []
for line in fp_val.readlines():
    filename_val, label_val, lesion_val = line.split()
    filenames_val.append(filename_val)
    if label_val == '0':
        mild_val.append(1)
        moderate_val.append(0)
        severe_val.append(0)
        verysevere_val.append(0)
    elif label_val == '1':
        mild_val.append(0)
        moderate_val.append(1)
        severe_val.append(0)
        verysevere_val.append(0)
    elif label_val == '2':
        mild_val.append(0)
        moderate_val.append(0)
        severe_val.append(1)
        verysevere_val.append(0)
    else:
        mild_val.append(0)
        moderate_val.append(0)
        severe_val.append(0)
        verysevere_val.append(1)
    lesions_val.append(lesion_val)
fp_val.close()

train_df = pd.DataFrame({
    'filename_train': filenames_train,
    'mild': mild_train,
    'moderate': moderate_train,
    'severe': severe_train,
    'verysevere': verysevere_train
})

val_df = pd.DataFrame({
    'filename_val': filenames_val,
    'mild': mild_val,
    'moderate': moderate_val,
    'severe': severe_val,
    'verysevere': verysevere_val
})

print(train_df.head())

total_train = train_df.shape[0]
total_val = val_df.shape[0]

fig,ax=plt.subplots(2,2,figsize=(20,20))
sns.barplot(y=train_df.mild.value_counts(),x=train_df.mild.value_counts().index,ax=ax[0,0])
ax[0,0].set_title("Value count for mild",size=13)
ax[0,0].set_xlabel('',size=13)
ax[0,0].set_ylabel('',size=13)

sns.barplot(y=train_df.moderate.value_counts(),x=train_df.moderate.value_counts().index,ax=ax[0,1])
ax[0,1].set_title("Value count for moderate",size=13)
ax[0,1].set_xlabel('',size=13)
ax[0,1].set_ylabel('',size=13)

sns.barplot(y=train_df.severe.value_counts(),x=train_df.severe.value_counts().index,ax=ax[1,0])
ax[1,0].set_title("Value count for severe",size=13)
ax[1,0].set_xlabel('',size=13)
ax[1,0].set_ylabel('',size=13)

sns.barplot(y=train_df.verysevere.value_counts(),x=train_df.verysevere.value_counts().index,ax=ax[1,1])
ax[1,1].set_title("Value count for verysevere",size=13)
ax[1,1].set_xlabel('',size=13)
ax[1,1].set_ylabel('',size=13)

inx =random.randint(0,1000)
img_sample = load_img(path_image + "/" + filenames_train[inx])
img_sample.show()

train_datagen = ImageDataGenerator(rotation_range= 10, width_shift_range=0.1, height_shift_range=0.1,brightness_range=[0.5, 1.5],
                                   shear_range=0.1, zoom_range=.1, fill_mode='nearest', rescale=1./255, horizontal_flip=True,
                                   vertical_flip=True)
train_generator = train_datagen.flow_from_dataframe(train_df, directory= path_image, target_size=(224, 224), x_col='filename_train',
                                                    y_col=["mild", "moderate", "severe", "verysevere"], class_mode='raw', batch_size= batch_size)

val_generator = train_datagen.flow_from_dataframe(val_df, directory=path_image, x_col= 'filename_val', y_col=["mild", "moderate", "severe", "verysevere"],
                                                  target_size=(224, 224), class_mode='raw', batch_size= batch_size_val)
history1 = Resnet50(train_generator, val_generator, epochs, batch_size, batch_size_val, total_train, total_val)

fig = go.Figure(data=[
    go.Line(name='Train_acc', x=history1.epoch, y=history1.history['accuracy']),
    go.Line(name='Validation_acc', x=history1.epoch, y=history1.history['val_accuracy'])
])
fig.update_layout(title="Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy", font=dict(family="Courier New, monospace", size=13, color="#7f7f7f"))
fig