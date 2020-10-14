import os
from shutil import copyfile
from random import randint
import random
from pathlib import Path
import functions
## define functions

## start code
mainPath = Path.cwd()
exPath = mainPath / "imagesAll/examples"
trainPath = mainPath / "images/train"
valPath = mainPath / "images/val"
mainPath = mainPath / "imagesAll"
# Delete all in examples folder
functions.emptyPath(exPath)
functions.emptyPath(trainPath)
functions.emptyPath(valPath)
# create directories
functions.createDir(exPath)
functions.createDir(trainPath)
functions.createDir(valPath)
# get main filenames
files = os.listdir(mainPath)
files = [f for f in os.listdir(mainPath) if not f.startswith('.')] # get all the image filenames without hidden ones
files = list(set(files)-set(['examples','train','val']))
numFiles = len(files)
# split data
# get 60% training data
numTrain = numFiles*.6
# get 20% validation data
numVal = numFiles*.2
# get 20% test data
numTest = numFiles*.2
# get random k number of targets
[files, trainList] = functions.splitImgs(files, numTrain)
[files, valList] = functions.splitImgs(files, numVal)
[files, testList] = functions.splitImgs(files, numTest)
# copy the files respectively
functions.copyFiles(exPath, mainPath, testList)
functions.copyFilesAge(valPath, mainPath, valList)
functions.copyFilesAge(trainPath, mainPath, trainList)
