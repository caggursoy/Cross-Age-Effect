## Cross-age effect Python executable code
import os
import matplotlib.pyplot as plt
import time
import numpy as np
import cv2
from random import randint
import datetime
import shutil

## First show the image then get the guesses
fileName = 'cae_gui.py'; # enter the name of the file
strPath = os.path.realpath(fileName); # get the path of the file
print("Please copy images folder to the following directory: ")
fileSiz = len(fileName); #filename size
pathSiz = len(strPath); # pathname size
dirName = strPath[0:(pathSiz-fileSiz)]; # dirName = '/' +strPath[1:(pathSiz-fileSiz)];
print(dirName)
imgDir = dirName +  'images'; # get directory
fName = raw_input('Please enter your first name: ');
sName = raw_input('Please enter your last name: ');
age = input('Please enter your age: ');
guessList = [];
fileList = [];
actualAgeList = [];
files = os.listdir(imgDir); # get all the image filenames
rng = range(0,25); # adjust range
for i in rng:
    idx = randint(0,len(imgDir)-1)
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
        actualAge = int(float(auxStr[auxSize-6:auxSize-4])); #-6,-4
        actualAgeList.append(actualAge);
print("Guessed")
print(guessList)
print("Actual")
print(actualAgeList)

## Print the results
vals = rng[-1]+1;
plt.plot(actualAgeList,'ro', guessList,'bx') # adjust this
plt.axis([-1, vals, -10, 80])
plt.ylabel('Ages')
plt.xlabel('Image index')
plt.legend()
plt.grid()
plt.show()

## Save the results to csv file
dateTime = str(datetime.datetime.now());
dateTime = dateTime[0:-7];
date = dateTime[0:-9];
outStr = 'Name: ' + fName + '\n' + 'Surname: ' + sName + '\n' + 'Age: ' + str(age) + '\n' + 'Date and Time: ' + dateTime + '\n' + 'Guessed ages: ' + str(guessList) + '\n' + 'Actual ages: ' + str(actualAgeList) + '\n' + 'Shown images: ' + str(fileList);
outFileName = dirName +'results\\'+fName+sName+dateTime+'.csv';
outFolderName = dirName +'results\\';
outFileName = outFileName[0:-15] + outFileName[-15:len(outFileName)].replace(":", "_"); #-15
if not os.path.exists(outFolderName): # If folder does not exist
    os.makedirs(outFolderName) # create folder
with open(outFileName,'wb') as file: # Create file and
    file.write(outStr) # write the results to csv

## Thanks and request folder
print("Thanks for your valuable effort. Would you kindly e-mail me the zipped file in the following folder: ")
print(dirName)
outputFileName = fName+sName
outDirName = dirName + 'results/'
shutil.make_archive(outputFileName, 'zip', outDirName)
