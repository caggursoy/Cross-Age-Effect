import os
from shutil import copyfile
from shutil import rmtree
import randomTargets

fileName = 'fileSort.py'; # enter the name of the file
# strPath = os.path.realpath(fileName); # get the path of the file
strPath = "/Volumes/Storage/Pythonfiles/caeMLage/imagesAll"
# newPath = strPath[0:-12] + "/images"
# strPath = strPath[0:-12] + "/imagesall";
newPath = "/Volumes/Storage/Pythonfiles/caeMLageMid/images"
if os.path.exists(newPath):
    rmtree(newPath)
os.mkdir(newPath)
# strPath = strPath[0:-12] + "/imagesall";
files = os.listdir(strPath); # get all the image filenames
for imgName in files:
    if imgName[0] == '.' or imgName == 'examples' or imgName == 'examples-copy':
        continue;
    auxSize = len(imgName);
    # print(imgName)
    start = imgName.index('male')+4;
    end = start + 2;
    if imgName[start] == "_" or imgName[start] == "+" :
        age = int(float(imgName[start+1:end+1]));
    else:
        age = int(float(imgName[start:end]));
    auxPathTrn = newPath  + "/" + str(age);
    if not os.path.exists(auxPathTrn): # If folder does not exist
        os.makedirs(auxPathTrn) # create folder
    src = strPath + "/" + imgName
    dstTrn = auxPathTrn + "/" + imgName
    copyfile(src, dstTrn)
