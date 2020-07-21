from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf

@tf.function
def Resnet50(train_generator, val_generator, epochs, batch_size, batch_size_val, total_train, total_val):
    model_finetuned = InceptionResNetV2(include_top = True, weights = 'imagenet', input_shape = (224,224,3))
    x = model_finetuned.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation= 'relu')(x)
    x = Dense(64, activation= 'relu')(x)
    pridictions = Dense(4, activation='softmax')(x)
    model_finetuned = Model(inputs = model_finetuned.input, outputs = pridictions)
    opt = Adam(learning_rate=0.001)
    model_finetuned.compile(optimizer=opt, loss='categorical_crossentropy', metrics= ['accuracy'])
    history0 = model_finetuned.fit_generator(train_generator, steps_per_epoch=total_train//batch_size, epochs= epochs,
                                            validation_data= val_generator, validation_steps=total_val//batch_size,
                                            verbose= 1, callbacks=[ReduceLROnPlateau(monitor='val_loss', factor= 0.5,
                                            patience=5, min_lr=0.000001)], use_multiprocessing= False, shuffle=True)
    model_finetuned.save('E:/Document/Do An III/Joint Acne Image Grading and Counting via Label Districbution Learning/model/resnet50.h5')
    model_finetuned.summary()
    return history0
