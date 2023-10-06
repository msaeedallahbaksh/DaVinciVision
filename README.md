# DaVinciVision 🎨
Aim is to build CNN prediction models which predict which artist created an artwork.

## File Explanation
- gitignore - contains file types to ignore for git.
- Image-Vis.mp4 - Video version of HSV Distribution of Van Gogh Painting
- TODO.md - contains broad next steps for the project.
- EDA - Directory containing Exploratory Data Analysis
    - Analysis.ipynb - Jupyter notebook overlooking Feature analysis for the NN
    - ImageDimensionResearch.ipynb - jupyter notebook entailing different resizing function.
- Model-Development
    - FilterGenerator.py - Custom Generator I built. Handles the insertion of new feature dimesions, data augmentation, class-weighting correction, one-hot-encoding, etc...
    - ModelTrain.py - Python Class which accepts numerous inputs ( model-architecture, hyperparameters, etc...) then has a function which allows you train the results and receive the feed back back as a dictionary. 
    - Tuning*.ipynb - rough draft of tuning & training notebooks. These notebooks basically just call the ModelTrain Class and train a series of models with given hyperparameters.
    - Results - Directory recording the training results. Each of the files inside is a pickle file in the form of a list, with each index containing a dictionary regarding a training session.

## Downloading Data
1. Pull the Repo. 
2. Download this dataset and **copy the directory where you saved it**.
3. Navigate to the *helpers/SetDataLocation.py* file and change the vlaue for the *path_to_dataset* variables with correct file path splitting for your os ( / - for windows )
4. Run the SetDataLocation.py File. Now you can work with the EDA noteobooks! 👏

## Dataset
https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time

| This dataset consists of three files. It is 2.32 GB in size which seems promising to train a quality DL model. There are a total of 8446 images (artworks) and 50 artists (classes). 

* artists.csv: A dataset containing information for each artist.
* images.zip: A collection of full-sized images, divided into folders and sequentially numbered.
* resized.zip: A resized version of the image collection, extracted from the folder structure for faster processing.

## EDA
Looking At HSV Visualization of Van Gogh Painting: 

![ezgif-4-12847aecd8](https://github.com/Charles-Gormley/DaVinciVision/assets/76138796/1693e2f9-f992-4fd6-9978-2d9b3ef45a0f)


## Models
### CNN Models 
* AlexNet
* VGG Network(s)
* NiN
* Inception Network(s)
* ResNet & ResNeXt
* DenseNet

## Class Path
- main-directory/
    - class_1/
        - image_1.jpg
        - image_2.jpg
        ...
    - class_2/
        - image_1.jpg
        - image_2.jpg
    etc...
- filtered-directory/
    - class_1-filtered/
        - image_1.jpg
        - image_2.jpg
        ...
    - class_2-filtered/
        - image_1.jpg
        - image_2.jpg
    etc...
etc...

## Results
#### BasicCNN - Initial Run
![image](https://github.com/Charles-Gormley/DaVinciVision/assets/76138796/0129f636-2542-419f-bd8f-b95151de6bf6)
#### BasicCNN - Tuning & Laplacian Filters
![image](https://github.com/Charles-Gormley/DaVinciVision/assets/76138796/627a1b01-3e61-4a1e-9cc1-58aea4a96265)
#### DenseNet - Tuning & Laplacian Filters
![image](https://github.com/Charles-Gormley/DaVinciVision/assets/76138796/99e49a47-8d02-4aa3-b4c2-a2d8dbff4c7f)
####  Resnet50 - Tuning & Transfer Learning
![image](https://github.com/Charles-Gormley/DaVinciVision/assets/76138796/f084f5ee-8b89-463c-9e0d-d2e20d5a0938)


