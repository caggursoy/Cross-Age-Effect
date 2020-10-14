# import packages
# import PlaidML first
import plaidml.keras
plaidml.keras.install_backend()
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the deepnet related packages
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import plot_model
from keras import backend as K
from keras.models import Model
import tensorflow as tf
from pyimagesearch.lenet import LeNet
from livelossplot import PlotLossesKeras
# import other packages
from imutils import paths
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import imutils
import os, random, cv2, argparse
from shutil import copyfile
import matplotlib.pyplot as plt
from shutil import copyfile
from random import randint
import functions
import platform
# define main
def main():
    inp0 = input('Assign images to groups? Y/N? ')
    if inp0.lower()=='y':
        import group_assign
    inp1 = input('Should I start training or go ahead to testing? Train/Test? ')
    if inp1.lower()=='train':
        verboseInfo = input('How detailed you want to see the training info? [(0): None, (1): Detailed, (2): Just epoch #]: ')
        functions.trainModel(EPOCHS = 500, INIT_LR = 1e-3, BS = 32, pxsize = 50,
            modelName='cae-model.model', plotName='model_perf.png', mainPath=Path.cwd(),
            trainPath='images/train', valPath='images/val', verboseInfo=verboseInfo)
    elif inp1.lower()=='test':
        modelName = input('Please enter the name of the saved model: ')
        functions.testModel(testPath='imagesAll/examples', pxsize = 50, modelName=modelName)

# init screen clearing function
def clear():
    if platform.system() == 'Linux' or platform.system() == 'Darwin':
        os.system('clear')
    else:
        os.system('cls')

# run main
if __name__ == '__main__':
    clear()
    print('Running main...')
    main()
