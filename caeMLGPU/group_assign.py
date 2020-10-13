import os
from shutil import copyfile
from random import randint
import random
from pathlib import Path
## define functions
def createDir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def emptyPath(path):
    # Delete all in the folder
    filelist = [ f for f in os.listdir(path) ]
    for f in filelist:
        os.remove(os.path.join(path, f))

def splitImgs(mainList, numElem):
    resList = random.sample(mainList,int(numElem))
    mainList = list(set(mainList)-set(resList))
    return [mainList, resList]

def copyFiles(dstPath, srcPath, list):
    for imgName in list: # copy the targets to target folder
        dst = dstPath / imgName
        src = srcPath / imgName
        copyfile(src, dst)

def copyFilesAge(dstPath, srcPath, list):
    for imgName in list: # copy the targets to target folder
        auxSize = len(imgName)
        start = imgName.index('male')+4
        end = start + 2
        if imgName[start] == "_" or imgName[start] == "+" :
            age = int(float(imgName[start+1:end+1]))
        else:
            age = int(float(imgName[start:end]))
        auxPathTrn = dstPath / str(age)
        if not os.path.exists(auxPathTrn): # If folder does not exist
            os.makedirs(auxPathTrn) # create folder
        dst = auxPathTrn / imgName
        src = srcPath / imgName
        copyfile(src, dst)

## start code
mainPath = Path.cwd()
exPath = mainPath / "imagesAll/examples"
trainPath = mainPath / "images/train"
valPath = mainPath / "images/val"
mainPath = mainPath / "imagesAll"
# create directories
createDir(exPath)
createDir(trainPath)
createDir(valPath)
# get main filenames
files = os.listdir(mainPath)
files = [f for f in os.listdir(mainPath) if not f.startswith('.')] # get all the image filenames without hidden ones
files = list(set(files)-set(['examples','train','val']))
numFiles = len(files)
# Delete all in examples folder
emptyPath(exPath)
emptyPath(trainPath)
emptyPath(valPath)
# split data
# get 60% training data
numTrain = numFiles*.6
# get 20% validation data
numVal = numFiles*.2
# get 20% test data
numTest = numFiles*.2
# get random k number of targets
[files, trainList] = splitImgs(files, numTrain)
[files, valList] = splitImgs(files, numVal)
[files, testList] = splitImgs(files, numTest)
# copy the files respectively
copyFiles(exPath, mainPath, testList)
copyFilesAge(valPath, mainPath, valList)
copyFilesAge(trainPath, mainPath, trainList)
