from tensorflow.keras.utils import Sequence
import cv2
import numpy as np
import os
from random import shuffle
import imgaug.augmenters as iaa
from tensorflow.keras.utils import to_categorical

from collections import defaultdict

def compute_custom_class_weight(labels):
    class_counts = np.bincount(labels)
    total_samples = np.sum(class_counts)
    class_weights = total_samples / (len(class_counts) * class_counts)
    return class_weights

class CustomGenerator(Sequence):
    def __init__(self, directory, batch_size, classes:list, laplacian=False, garbor=False, augments=False, weighted=False, image_size=225):
        self.directory = directory
        self.batch_size = batch_size
        self.laplacian = laplacian
        self.garbor = garbor
        self.augments = augments
        self.classes = classes
        self.oversample = weighted
        self.im_size = image_size

        self.augmenter = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.SaltAndPepper(0.1), # Noise Injection
            iaa.GaussianBlur(sigma=(0.0, 3.0))
        ])
        self.class_names = self.classes
        self.class_mapping = {class_name: i for i, class_name in enumerate(self.class_names)}
        self.file_list = []
        self.labels = []
        for class_name in self.class_names:
            class_path = os.path.join(directory, class_name)
            files = os.listdir(class_path)
            self.file_list.extend([os.path.join(class_path, file) for file in files])
            self.labels.extend([self.class_mapping[class_name]] * len(files))
        combined = list(zip(self.file_list, self.labels))
        shuffle(combined)
        self.file_list, self.labels = zip(*combined)
        self.labels_categorical = to_categorical(self.labels, num_classes=(len(self.class_names)))

        # Compute class weights
        class_weights = compute_custom_class_weight(self.labels)
        self.class_weights = dict(zip(np.unique(self.labels), class_weights))

        # Calculate number of samples per class for oversampling
        self.samples_per_class = defaultdict(int)
        for label in self.labels:
            self.samples_per_class[label] += 1

    def load_and_preprocess_image(self, image_path):
        # Strip image path
        image = cv2.imread(image_path)

        # Preprocess the image as needed
        # Example: Convert color channels (e.g., BGR to RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (self.im_size, self.im_size))
        preproccesed_image = np.array(image_rgb)


        # Image pathing
        sub_image_path = os.path.join(*(image_path.split(os.sep))[-2:])

        if self.laplacian:
            base_lap = "/home/ceg98/Documents/Laplacian/"
            lap_path = base_lap  + sub_image_path
            
            lap_image = cv2.imread(lap_path, cv2.IMREAD_GRAYSCALE)
            lap_image = cv2.resize(lap_image, (self.im_size, self.im_size))
            preproccesed_image = np.dstack((preproccesed_image, np.array(lap_image)))
            
        
        if self.garbor:
            base_garbor = "/home/ceg98/Documents/Garbor/"
            garbor_path = base_garbor + sub_image_path
            
            garbor_image = cv2.imread(garbor_path)
            garbor_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            garbor_image = cv2.resize(garbor_image, (self.im_size, self.im_size))
            preproccesed_image = np.dstack((preproccesed_image, np.array(garbor_image)))
        
        
        return preproccesed_image
    
    def oversample_minority_classes(self, batch_files, batch_labels):
        oversampled_files = []
        oversampled_labels = []
        for file, label in zip(batch_files, batch_labels):
            oversampled_files.append(file)
            oversampled_labels.append(label)
            if self.samples_per_class[label] < self.batch_size:
                oversampled_files.extend([file] * (self.batch_size - self.samples_per_class[label]))
                oversampled_labels.extend([label] * (self.batch_size - self.samples_per_class[label]))
        return oversampled_files, oversampled_labels
        

    def __getitem__(self, index):
        batch_files = self.file_list[index * self.batch_size : (index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size : (index + 1) * self.batch_size]
        if self.oversample:
            batch_files, batch_labels = self.oversample_minority_classes(batch_files, batch_labels)

        categorical_labels = self.labels_categorical[index * self.batch_size : (index + 1) * self.batch_size]

        batch_images = []
        i = 0
        for file_name in batch_files:
            image_path = os.path.join(self.directory, file_name)
            # Load the image and perform any necessary preprocessing
            image = self.load_and_preprocess_image(image_path)
            batch_images.append(image)
            i += 1

        
        # Convert the list of images to a numpy array
        batch_images = np.array(batch_images)
        
         # Augment the images
        if self.augments:
            batch_images = self.augmenter.augment_images(batch_images)

        return batch_images, categorical_labels

    def __len__(self):
        return len(self.file_list) // self.batch_size