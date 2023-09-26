######### Importing #########
#  Data Science Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#  Machine Learning Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Internal Libraries
import json
import os
from tqdm import tqdm, tqdm_notebook
import random
import shutil

class TrainModel:
    def __init__(self, architecture:str, batch_size:int, image_size:int, validation_split:float, learning_rate:float, seed_n:int, verbose:int, best_classes:bool, normalization:bool, dropout:bool, dropout_rate:float, hsv:bool, garbor:bool, laplacian:bool, home_dir="/home/ceg98/Documents/"):
        tf.random.set_seed(seed_n)
        
        self.n_features = 3
        self.seed_n = seed_n
        self.archictecture = architecture
        self.batch_size = batch_size
        self.image_size = image_size
        self.validation_split = validation_split
        self.lr = learning_rate
        self.images_dir = home_dir + "archive/resized/resized"
        self.train_input_shape = (self.image_size, self.image_size, self.n_features)
        self.home_dir = home_dir
        self.verbose = verbose
        self.best_classes = best_classes
        self.normalization = normalization
        self.dropout = dropout
        self.dropout_rate = dropout_rate

        # Extra Filters
        self.hsv = hsv
        self.garbor = garbor
        self.laplacian = laplacian

        # VGG
        self.blocks_of_vgg = 11

        # DenseNet
        self.blocks_per_layer = [6, 12, 24, 16] # DenseNet-121 is base.
            # DenseNet-121: [6, 12, 24, 16]
            # DenseNet-169: [6, 12, 32, 32]
            # DenseNet-201: [6, 12, 48, 32]
            # DenseNet-264: [6, 12, 64, 48]
        self.growth_rate = 32
        self.reduction = 0.5
        

    def format_data(self):

        # TODO: This function needs serious cleaning.
        df = pd.read_csv(self.home_dir + "archive/artists.csv")

        if self.best_classes:
            artists_top = df[df['paintings'] >= 200].reset_index()
        else:
            artists_top = df[df['paintings'] >= 1].reset_index()
        artists_top = artists_top[['name', 'paintings']]
        artists_top['class_weight'] = max(artists_top.paintings)/artists_top.paintings
        artists_top['class_weight'] = artists_top.paintings.sum() / (artists_top.shape[0] * artists_top.paintings)
        artists_top['weights'] = artists_top['class_weight']
        self.weighted_artists = artists_top

        self.class_weights = self.weighted_artists['weights'].to_dict() # TODO: Check if this causing a bug. 
        self.artists_name = self.weighted_artists['name'].str.replace(' ', '_').values
        self.images_dir = self.home_dir + "archive/resized/resized"
        self.n_classes = self.weighted_artists.shape[0]

    def create_generators(self):
        self.train_datagen = ImageDataGenerator(validation_split=0.2,
                                                rescale=1/255,
                                                rotation_range=45,
                                                zoom_range=0.7,
                                                horizontal_flip=True,
                                                vertical_flip=True
                                                )
        
        self.train_generator = self.train_datagen.flow_from_directory(directory=self.images_dir,
                                                            class_mode='categorical',
                                                            target_size=self.train_input_shape[0:2],
                                                            batch_size=self.batch_size,
                                                            subset="training",
                                                            shuffle=True,
                                                            classes=self.artists_name.tolist()
                                                        )

        self.valid_generator = self.train_datagen.flow_from_directory(directory=self.images_dir,
                                                            class_mode='categorical',
                                                            target_size=self.train_input_shape[0:2],
                                                            batch_size=self.batch_size,
                                                            subset="validation",
                                                            shuffle=True,
                                                            classes=self.artists_name.tolist()
                                                        )
        
        self.STEP_SIZE_TRAIN = self.train_generator.n//self.train_generator.batch_size
        self.STEP_SIZE_VALID = self.valid_generator.n//self.valid_generator.batch_size
    
    # VGG Block Functions
    def vgg_block(self, num_filters, num_layers):
        for _ in range(num_layers):
            self.model.add(Conv2D(num_filters, kernel_size=(3, 3), padding='same', activation='relu'))
            if self.dropout:
                self.model.add(Dropout(self.dropout_rate))
        if self.normalization:
            self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    def vgg_model(self):
        self.model = Sequential()

        # VGG configurations
        vgg_configs = {
            11: [1, 1, 2, 2, 2],
            13: [2, 2, 2, 2, 2],
            16: [2, 2, 3, 3, 3],
            19: [2, 2, 4, 4, 4]
        }

        assert self.blocks_of_vgg in vgg_configs.keys(), f"Number of blocks not supported for VGG. Choose from {vgg_configs.keys()}."

        layers = vgg_configs[self.blocks_of_vgg]

        for i, num_layers in enumerate(layers):
            self.vgg_block((64*(2**i)), num_layers)

        self.model.add(Flatten())
        self.model.add(Dense(4096, activation='relu'))
        if self.normalization:
            self.model.add(BatchNormalization())
        if self.dropout:
            self.model.add(Dropout(self.dropout_rate))

        self.model.add(Dense(4096, activation='relu'))
        if self.normalization:
            self.model.add(BatchNormalization())
        if self.dropout:
            self.model.add(Dropout(self.dropout_rate))

        self.model.add(Dense(self.n_classes, activation='softmax'))

    def get_architecture(self):
        if self.archictecture == "ResNet50":
            self.base_model = ResNet50(weights='imagenet', 
                                       include_top=False, 
                                       input_shape=self.train_input_shape,
                                       batch_norm = self.normalization)
            self.transfer_learning = True

        elif self.archictecture == "BasicCNN":
            self.model = Sequential()
            self.model.add(Conv2D(32, (1, 1), input_shape=self.train_input_shape))
            if self.normalization:
                self.model.add(BatchNormalization())
            if self.dropout:
                self.model.add(Dropout(self.dropout_rate))
            self.model.add(MaxPooling2D(pool_size=2, strides=2))

            self.model.add(Conv2D(64, (1, 1), activation='relu', padding='same'))
            if self.normalization:
                self.model.add(BatchNormalization())
            if self.dropout:
                self.model.add(Dropout(self.dropout_rate))
            self.model.add(MaxPooling2D(pool_size=2, strides=2))

            self.model.add(Conv2D(128, (1, 1), activation='relu', padding='same'))
            if self.normalization:
                self.model.add(BatchNormalization())
            if self.dropout:
                self.model.add(Dropout(self.dropout_rate))
            self.model.add(MaxPooling2D(pool_size=2, strides=2))

            self.model.add(Conv2D(256, (1, 1), activation='relu', padding='same'))
            if self.normalization:
                self.model.add(BatchNormalization())
            if self.dropout:
                self.model.add(Dropout(self.dropout_rate))
            self.model.add(MaxPooling2D(pool_size=2, strides=2))

            self.model.add(Conv2D(512, (1, 1), activation='relu', padding='same'))
            if self.normalization:
                self.model.add(BatchNormalization())
            if self.dropout:
                self.model.add(Dropout(self.dropout_rate))
            self.model.add(MaxPooling2D(pool_size=2, strides=2))

            self.model.add(Flatten())
            self.model.add(Dense(1024, activation='tanh'))
            if self.normalization:
                self.model.add(BatchNormalization())
            if self.dropout:
                self.model.add(Dropout(self.dropout_rate))


            self.model.add(Dense(self.n_classes, activation='softmax'))

            self.transfer_learning = False
        
        elif self.archictecture == 'AlexNet':
            self.model = Sequential()
            self.model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=self.train_input_shape))
            if self.normalization:
                self.model.add(BatchNormalization())
            self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

            self.model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"))
            if self.normalization:
                self.model.add(BatchNormalization())
            self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

            self.model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"))
            self.model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"))
            self.model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"))
            if self.normalization:
                self.model.add(BatchNormalization())
            self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))


            self.model.add(Flatten())


            self.model.add(Dense(4096, activation='relu'))
            if self.normalization:
                self.model.add(BatchNormalization())

            self.model.add(Dense(4096, activation='relu'))
            if self.normalization:
                self.model.add(BatchNormalization())

            self.model.add(Dense(1000, activation='relu'))
            if self.normalization:
                self.model.add(BatchNormalization())

            self.model.add(Dense(self.n_classes, activation='softmax'))

            self.transfer_learning = False
        
        elif self.archictecture == 'VGG':
            self.vgg_model() # This will create the self.model variable
            self.transfer_learning = False

        elif self.archictecture == 'DenseNet':
            self.transfer_learning = False
            self.model = DenseNet121(include_top=True,
                                     weights=None,
                                     input_tensor=None,
                                     input_shape=self.train_input_shape,
                                     pooling=None,
                                     classes=self.n_classes,
                                     classifier_activation="softmax"
                                    )

        elif self.archictecture == "GoogLeNet":
            self.transfer_learning = False
            self.model = InceptionV3(include_top=True,
                                     weights=None,
                                     input_tensor=None,
                                     input_shape=self.train_input_shape,
                                     pooling=None,
                                     classes=self.n_classes,
                                     batch_norm=self.normalization
                                    )

    def define_architecture(self):
        
        if self.transfer_learning: 
            for layer in self.base_model.layers:
                layer.trainable = True

            # Add layers at the end
            X = self.base_model.output
            X = Flatten()(X)

            X = Dense(512, kernel_initializer='he_uniform')(X)
            X = BatchNormalization()(X)
            X = Activation('relu')(X)

            X = Dense(16, kernel_initializer='he_uniform')(X)
            X = BatchNormalization()(X)
            X = Activation('relu')(X)

            output = Dense(self.n_classes, activation='softmax')(X)

            self.model = Model(inputs=self.base_model.input, outputs=output)

    def short_model(self):
        n_epoch = 2
        self.early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, 
                           mode='auto', restore_best_weights=True)

        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, 
                                    verbose=1, mode='auto')

        optimizer = Adam(lr=self.lr)
        self.model.compile(loss='categorical_crossentropy',
                    optimizer=optimizer, 
                    metrics=['accuracy'])


        self.short_history = self.model.fit_generator(generator=self.train_generator, 
                                                      steps_per_epoch=self.STEP_SIZE_TRAIN,
                                                      validation_data=self.valid_generator, 
                                                      validation_steps=self.STEP_SIZE_VALID,
                                                      epochs=n_epoch,
                                                      shuffle=True,
                                                      verbose=self.verbose,
                                                      callbacks=[self.reduce_lr],
                                                      use_multiprocessing=True,
                                                      workers=16,
                                                      class_weight=self.class_weights
                                                     )
    
    def full_model(self):
        if self.archictecture == "ResNet50":
            # Freeze core ResNet layers and train again 
            model = self.model
            for layer in model.layers:
                layer.trainable = False

            for layer in model.layers[:50]:
                layer.trainable = True

            self.model = model

        
        optimizer = Adam(lr=self.lr)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer, 
                           metrics=['accuracy'])

        n_epoch = 2
        self.full_history = self.model.fit_generator(generator=self.train_generator, 
                                                     steps_per_epoch=self.STEP_SIZE_TRAIN,
                                                     validation_data=self.valid_generator, 
                                                     validation_steps=self.STEP_SIZE_VALID,
                                                     epochs=n_epoch,
                                                     shuffle=True,
                                                     verbose=self.verbose,
                                                     callbacks=[self.reduce_lr, self.early_stop],
                                                     use_multiprocessing=True,
                                                     workers=16,
                                                     class_weight=self.class_weights
                                                    )
        
        


    def train(self):
        self.format_data()
        self.create_generators()
        self.get_architecture()
        self.define_architecture()
        self.short_model()
        
        # Saving Training Data:
        history = {}
        history['loss'] = self.short_history.history['loss'] 
        history['accuracy'] = self.short_history.history['accuracy'] 
        history['val_loss'] = self.short_history.history['val_loss'] 
        history['val_accuracy'] = self.short_history.history['val_accuracy'] 
        history['lr'] = self.short_history.history['lr']

        history['last-accuracy'] = history['accuracy'][-1]
        history['last-loss'] = history['loss'][-1]
        history['last-val_loss'] = history['val_loss'][-1]
        history['last-val_accuracy'] = history['val_accuracy'][-1]
        history['last-lr'] = history['lr'][-1]

        # Hyperparams selected 
        history['arch'] = self.archictecture
        history['batch-size'] = self.batch_size
        history['image-size'] = self.image_size
        history['learning-rate'] = self.lr
        history['seed'] = self.seed_n
        history['valid_split'] = self.validation_split

        if history['last-accuracy'] > 0.14: # Making sure first couple layers is atleast above 50% accuracy.
            self.full_model()

            history['loss'] += self.full_history.history['loss']
            history['accuracy'] += self.full_history.history['accuracy']
            history['val_loss'] += self.full_history.history['val_loss']
            history['val_accuracy'] += self.full_history.history['val_accuracy']
            history['lr'] += self.full_history.history['lr']

            history['last-accuracy'] = history['accuracy'][-1]
            history['last-loss'] = history['loss'][-1]
            history['last-val_loss'] = history['val_loss'][-1]
            history['last-val_accuracy'] = history['val_accuracy'][-1]
            history['last-lr'] = history['lr'][-1]
        
        return history
