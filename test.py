from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pandas as pd
import numpy as np

def main(testimages):
    model = load_model('model/resnet50_dro.h5', custom_objects=None, compile=False)
    filenames = os.listdir(testimages)
    test_df = pd.DataFrame({'filename': filenames})
    batch_size_test = 1

    test_datagen = ImageDataGenerator(rotation_range= 10, width_shift_range=0.1, height_shift_range=0.1,brightness_range=[0.5, 1.5],
                                       shear_range=0.1, zoom_range=.1, fill_mode='nearest', rescale=1./255, horizontal_flip=True,
                                       vertical_flip=True)
    test_generator = test_datagen.flow_from_dataframe(test_df, directory= testimages , target_size=(224,224), x_col="filename",
                                                      y_col=None, class_mode=None, shuffle= False, batch_size=batch_size_test)
    count = len(filenames)
    preds = model.predict_generator(test_generator, steps=count)
    probs = []
    for i in range(count):
        pred = np.argmax(preds[i])
        probs.append(pred)
    test_df['level'] = probs
    plt.figure(figsize=(12,8))
    for index,row in test_df.iterrows():
        filename = row['filename']
        level = row['level']
        plt.subplot(count//3,3,index+1)
        img = mpimg.imread(testimages + '/' +filename)
        plt.imshow(img)
        plt.axis('On')
        if level==0:
            plt.xlabel("(" + filename + ")" + 'mild')
        elif level==1:
            plt.xlabel("(" + filename + ")" + "moderate")
        elif level==2:
            plt.xlabel("(" + filename + ")" + "severe")
        else:
            plt.xlabel("(" + filename + ")" + "very severe")

    plt.tight_layout()
    plt.show()