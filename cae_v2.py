# -*- coding: utf-8 -*-
## Cross-age effect Python executable code using Park Aging Mind Lab at UTDallas
import os
import matplotlib.pyplot as plt
import time
import numpy as np
import cv2
from random import randint
import datetime
import shutil

##
fileName = 'cae_v2.py'; # enter the name of the file
strPath = os.path.realpath(fileName); # get the path of the file
imgDir = '/Volumes/Storage/Face datasets/FaceDatabase/Neutral Faces/neutral_bmp'; # get directory
fileSiz = len(fileName); #filename size
pathSiz = len(strPath); # pathname size
dirName = '/' +strPath[1:(pathSiz-fileSiz)];
print(dirName)
fName = raw_input('Please enter your first name: ');
sName = raw_input('Please enter your last name: ');
age = input('Please enter your age: ');
guessList = [];
fileList = [];
actualAgeList = [];
files = os.listdir(imgDir); # get all the image filenames
rng = range(0,2); # adjust range (25)
for i in rng:
    idx = randint(0,len(files)-1)
    img = imgDir+'/'+files[idx];
    imgMat = cv2.imread(img);
    auxName = files[idx];
    auxStr = auxName;
    while auxName not in fileList: # still shows same image
        fileList.append(auxName)
        winname = 'Images'
        cv2.namedWindow(winname)        # Create a named window
        cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
        cv2.imshow(winname, imgMat)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
        ageGuess = input('Please enter your age guess: ')
        guessList.append(ageGuess);
        auxSize = len(auxStr);
        start = auxStr.index('male')+4;
        end = start + 2;
        if auxStr[start] == "_" or auxStr[start] == "+" :
            actualAge = int(float(auxStr[start+1:end+1]));
        else:
            actualAge = int(float(auxStr[start:end]));
        actualAgeList.append(actualAge);
print("Guessed")
print(guessList)
print("Actual")
print(actualAgeList)

## Print the results
# vals = rng[-1]+1;
# plt.plot(actualAgeList,'ro', guessList,'bx') # adjust this
# plt.axis([-1, vals, -10, 80])
# plt.ylabel('Ages')
# plt.xlabel('Image index')
# plt.legend()
# plt.grid()
# plt.show()

## Save the results to csv file
dateTime = str(datetime.datetime.now());
dateTime = dateTime[0:-7];
date = dateTime[0:-9];
outStr = 'Name: ' + fName + '\n' + 'Surname: ' + sName + '\n' + 'Age: ' + str(age) + '\n' + 'Date and Time: ' + dateTime + '\n' + 'Guessed ages: ' + str(guessList) + '\n' + 'Actual ages: ' + str(actualAgeList) + '\n' + 'Shown images: ' + str(fileList);
outFileName = '/'+strPath[1:(pathSiz-fileSiz)] +'results/'+fName+sName+dateTime+'.csv';
outFolderName = '/'+strPath[1:(pathSiz-fileSiz)] +'results/';
print(outFileName)
# outFileName = outFileName.replace(" ", "_");
# outFileName = outFileName.replace(":", "_");
if not os.path.exists(outFolderName): # If folder does not exist
    os.makedirs(outFolderName) # create folder
with open(outFileName,'wb') as file: # Create file and
    file.write(outStr) # write the results to csv

## Thanks and request folder
print("Thanks for your valuable effort. Would you kindly e-mail me the zipped file in the following folder: ")
print(dirName)
outputFileName = fName+sName
outDirName = dirName + 'results/'
# shutil.make_archive(outputFileName, 'zip', outDirName) # for local purposes zipping is unnecessary
