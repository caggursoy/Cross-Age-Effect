# -*- coding: utf-8 -*-
import csv
import os
from random import randint
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
##
fileName = 'dataReader.py'; # enter the name of the file
strPath = os.path.realpath(fileName); # get the path of the file
fileSiz = len(fileName); #filename size
pathSiz = len(strPath); # pathname size
dirName = '/' +strPath[1:(pathSiz-fileSiz)];
imgDir = dirName +  'All'; # get directory
files = os.listdir(imgDir); # get all the image filenames
genList = []; invList = []; nameList = []; ageList = [];
guessList = []; actualList = []; imageList = []; imgsList = [];
imgsFreqs = []; imgsFreqSorted = []; mostFreqImgs = []; freqList = [];
mostFreqImNames = []; freqListSorted = []; plotImgs =[]; plotNames = [];
plotAges = []; plotActuals = []; plotGuesses = [];
for qq in files:
    if qq[0] == ".":
        files.remove(qq);
k=0;
for ff in files:
    toRead = 'All/' + ff;
    # print(ff)
    with open(toRead, 'rb') as csvfile:
        fileRead = csv.reader(csvfile, delimiter='\n', quotechar='|')
        for row in fileRead:
            # print(row)
            invList.append(row)
    genList.append(invList[(0+(7*k)):(7+(7*k))]) # row range: 0:7
    k=k+1;
genList.append(invList)
mm = genList[0]
for ll in genList:
    name = ''.join(map(str, ll[0]))
    name = name[6:len(name)]
    surname = ''.join(map(str, ll[1]))
    surname = surname[9:len(surname)]
    nameList.append(name+' '+surname)
    age = ''.join(map(str, ll[2]))
    age = int(float(age[5:len(age)]))
    ageList.append(age)
    guess = ''.join(map(str, ll[4]))
    guess = guess[14:len(guess)]
    guess = guess.replace("[","")
    guess = guess.replace("]","")
    guess = guess.replace(" ","")
    subGuess = guess.split(",")
    subGuess = map(float, subGuess) 
    guessList.append(subGuess)
    actual = ''.join(map(str, ll[5]))
    actual = actual[13:len(actual)]
    actual = actual.replace("[","")
    actual = actual.replace("]","")
    actual = actual.replace(" ","")
    subActual = map(int, actual.split(","))
    actualList.append(subActual)
    imgs = ''.join(map(str, ll[6]))
    imgs = imgs[14:len(imgs)]
    imgs = imgs.replace("[","")
    imgs = imgs.replace("]","")
    imgs = imgs.replace(" ","")
    imgs = imgs.replace("'","")
    imageList.append(imgs.split(","))
ii = 5;
for nn in imageList:
  for ll in nn:
      imgsList.append(ll)
cnt = 0;
for ww in imgsList:
    for ij in range(0,len(imgsList)-1):
        if ww == imgsList[ij]:
            cnt = cnt + 1;
    imgsFreqs.append([ww, cnt])
    cnt = 0
imgsFreqs = (sorted(imgsFreqs, key=itemgetter(1)));
for rr in imgsFreqs:
    if rr not in imgsFreqSorted:
        imgsFreqSorted.append(rr)
numOfFreqImgs = 5 # number of most frequent images
mostFreqImgs = list(reversed(imgsFreqSorted[len(imgsFreqSorted)-numOfFreqImgs:len(imgsFreqSorted)]))
for nn in mostFreqImgs:
    mostFreqImNames.append(nn[0])
indices = len(imageList)
for ids in range(0,indices-1):
    for rr in range(0,len(imageList[ids])-1):
        qq = imageList[ids]
        if qq[rr] in mostFreqImNames:
            freqList.append((qq[rr],nameList[ids],ageList[ids],(actualList[ids])[rr],(guessList[ids])[rr]))
freqListSorted = (sorted(freqList, key=itemgetter(0)));
# print(freqListSorted[0][0])
for nx in freqListSorted:
    plotImgs.append(nx[0]) # image names
    plotNames.append(nx[1]) # participant names
    plotAges.append(nx[2]) # participant ages, rx
    plotActuals.append(nx[3]) # actual ages, bo
    plotGuesses.append(nx[4]) # guessed ages, go
cntOfUnq=1; # count of unique images (in order to check if it's really 20)
aux = (freqListSorted[0])[0]
ind2 = [] # indices where the image changes
cntt = 0;
for oo in freqListSorted:
    if aux != oo[0]:
        cntOfUnq=cntOfUnq+1
        aux = oo[0]
        ind2.append(cntt)
    cntt=cntt+1
ind2.insert(0,0)
for tt in range(0,cntOfUnq-1):
  plt.figure(figsize=(10,8))
  plt.plot(plotAges[ind2[tt]:ind2[tt+1]],'rx')
  plt.plot(plotActuals[ind2[tt]:ind2[tt+1]],'bo')
  plt.plot(plotGuesses[ind2[tt]:ind2[tt+1]],'go')
  plt.axis([-1, len(plotAges[ind2[tt]:ind2[tt+1]]), -10, 80])
  plt.title(plotImgs[ind2[tt]])
  plt.ylabel('Ages')
  plt.xlabel('Trial index')
  plt.legend(('Age','Actual', 'Guessed')) # loc='upper right'
  plt.grid()
  plt.show()
