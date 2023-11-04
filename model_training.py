import os
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout,Input,Flatten,Dense,MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping, ReduceLROnPlateau

BATCH_SIZE = 4
EPOCHS = 2

train_the_datagenerator = ImageDataGenerator(rescale = 1./255, rotation_range = 0.2,shear_range = 0.2,
                                            zoom_range = 0.2,width_shift_range = 0.2,
                                            height_shift_range = 0.2, validation_split = 0.2)

train_the_data= train_the_datagenerator.flow_from_directory(os.path.join('After_separation', 
                                                                        'Few_closed_images', 'train'),
                                target_size = (80,80), batch_size = BATCH_SIZE, 
                                class_mode = 'categorical',subset='training' )

validation_data= train_the_datagenerator.flow_from_directory(os.path.join('After_separation',
                                                                        'Few_opened_images', 'train'),
                                target_size = (80,80), batch_size = BATCH_SIZE, 
                                class_mode = 'categorical', subset='validation')


test_the_datagenerator = ImageDataGenerator(rescale = 1./255)

test_the_data = test_the_datagenerator.flow_from_directory(os.path.join('After_separation', 'Few_closed_images',
                                                                        'Few_opened_images','test'),
                                target_size=(80,80), batch_size = BATCH_SIZE, class_mode='categorical')



b_model = InceptionV3(include_top = False, weights = 'imagenet', 
                    input_tensor = Input(shape = (80,80,3)))
h_model = b_model.output
h_model = Flatten()(h_model)
h_model = Dense(64, activation = 'relu')(h_model)
h_model = Dropout(0.5)(h_model)
h_model = Dense(2,activation = 'softmax')(h_model)

model = Model(inputs = b_model.input, outputs= h_model)
for layer in b_model.layers:
    layer.trainable = False


check_point = ModelCheckpoint(os.path.join("models", "model.h5"),
                            monitor = 'val_loss', save_best_only = True, verbose = 3)

early_stop = EarlyStopping(monitor = 'val_loss', patience = 7, 
                        verbose= 3, restore_best_weights = True)


learning_rate = ReduceLROnPlateau(monitor= 'val_loss', patience=3, verbose= 3, )

call_backs = [check_point, early_stop, learning_rate]



model.compile(optimizer = 'Adam', 
            loss = 'categorical_crossentropy', 
            metrics = ['accuracy'])


model.fit_generator(train_the_data,steps_per_epoch = train_the_data.samples// BATCH_SIZE,
                validation_data = validation_data,
                validation_steps = validation_data.samples// BATCH_SIZE,
                callbacks = call_backs,
                    epochs = EPOCHS)


# Model Evaluation

acc_tr, loss_tr = model.evaluate_generator(train_the_data)
print(acc_tr)
print(loss_tr)

acc_vr, loss_vr = model.evaluate_generator(validation_data)
print(acc_vr)
print(loss_vr)

acc_test, loss_test = model.evaluate_generator(test_the_data)
print(acc_tr)
print(loss_tr)