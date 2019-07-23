import plaidml.keras
# import PlaidML
plaidml.keras.install_backend()
# import necessary packages
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array
from sklearn import datasets, svm, metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import cv2
import os
##
imgFolder = os.listdir('images/')
if '.DS_Store' in imgFolder:
  imgFolder.remove('.DS_Store')
labels = imgFolder # set labels
## define functions
# show images
def showIm(imgArr,savFig=False,savNm='fig1.png',imgTitle='',imgLegend=''):
    fig, ax = plt.subplots()
    if len(imgArr.shape) > 3:
        arrToShow = np.squeeze(imgArr)
    else:
        arrToShow = imgArr
    ax.imshow(arrToShow)
    if savFig:
        plt.savefig(savNm)
    else:
        fig.canvas.draw()
        plt.title(imgTitle)
        ax.legend(imgLegend)
        plt.show()
# unique elements in lists
# function to get unique values
def unique(list1,show=False):
    # intilize a null list
    unique_list = []
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
    if show:
        for x in unique_list:
            print(x,',')
    return unique_list
## end functions

## dummy trial for img sizes
imgPaths = 'images/'
idx=0
pxsize = 200
for p in labels:
    aaa = imgPaths + p # set image paths for each folder
    imgDest = os.listdir(aaa)
    for ii in imgDest:
        idx+=1
        # image = cv2.imread(aaa+'/'+ii) # read images
        image = plt.imread(aaa+'/'+ii) # read images
        image = cv2.resize(image, (pxsize, pxsize))
        imgShape = image.shape

## actual work here
allImgs = np.empty([idx,imgShape[0],imgShape[1],imgShape[2]]).astype('uint8')
idx2=0
imgNames = []
for p in labels:
    imgAuxPaths = imgPaths + p # set image paths for each folder
    imgDest = os.listdir(imgAuxPaths)
    for ii in imgDest:
        imgNames.append(ii)
        image = plt.imread(imgAuxPaths+'/'+ii).astype('uint8') # read images
        image = cv2.resize(image, (pxsize, pxsize))
        if image.shape == imgShape:
            allImgs[idx2,:,:,:] = image
        else:
            if image.shape[0] != imgShape[0]:
                bb = np.empty([abs(image.shape[0]-imgShape[0]),image.shape[1],image.shape[2]]).astype('uint8')
                # print('bb:',bb.shape,'image:',image.shape)
                image = np.squeeze(np.append([image],[bb],axis=1))
                # print('padded image:',image.shape)
            allImgs[idx2,:,:,:] = image
        idx2=idx2+1

# get targets
tList = []
for nm in imgNames:
    try:
        tAge = int(nm[nm.index('male')+4:nm.index('male')+6])
    except ValueError:
        tAge = int(nm[nm.index('male')+5:nm.index('male')+7])
    tList.append(tAge)
allTargets = np.asarray(tList)
targets = np.asarray(unique(tList))

## init training
# flatten all
allImgsFl = np.empty([580,(200*200*3)]).astype('uint8')
allTargetsFl = np.empty([580])
for pp in range(580):
    allImgsFl[pp] = allImgs[pp,:,:,:].flatten()
    allTargetsFl[pp] = allTargets[pp].flatten()

# Train - test split
(trainX, testX, trainY, testY) = train_test_split(allImgsFl,
	allTargets, test_size=0.2, random_state=42)

cv = StratifiedKFold(n_splits=6)
classifier = svm.SVC(kernel='linear', probability=True, random_state=42)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
X = allImgsFl
y = allTargets
for train, test in cv.split(X, y):
    yBin = label_binarize(y[test], classes=range(18,94))
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(yBin.argmax(axis=1), probas_[:, 1], pos_label=1)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = metrics.auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = metrics.auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristics')
plt.legend(loc="lower right")
plt.grid()
plt.savefig('svm_roc1.png', dpi=800)
plt.show()

predY = classifier.predict(testX)
np.set_printoptions(threshold=sys.maxsize)
cm = metrics.confusion_matrix(testY, predY,labels=range(18,94))
print("Confusion matrix:\n%s" % cm, file=open('confMatrixAllGroup.txt', 'w'))
print(metrics.classification_report(testY, predY))
plt.imshow(cm,cmap='binary')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion matrix')
plt.xticks(np.arange(0, 76, step=5),np.arange(18, 94, step=5))
plt.yticks(np.arange(0, 76, step=5),np.arange(18, 94, step=5))
plt.xlim([-0.5, 75.5])
plt.ylim([-0.5, 75.5])
plt.colorbar()
plt.savefig('svm_cm_Allgroup.png',dpi=800)
plt.show()
