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
from shutil import copyfile, rmtree
from random import randint
###
from keras.callbacks import LambdaCallback
from IPython.display import clear_output
import pickle
import matplotlib.pyplot as plt
###
## define functions
def createDir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def emptyPath(path):
    # Delete all in the folder
    if os.path.isdir(path):
        rmtree(path)
    # filelist = [ f for f in os.listdir(path) ]
    # for f in filelist:
    #     os.remove(os.path.join(path, f))

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

def plot_metrics( epoch, ep_metrics ):
# source: https://www.reddit.com/r/MachineLearning/comments/65jelb/d_live_loss_plots_inside_jupyter_notebook_for/
    logs = pickle.load(open( LOGS_PATH, 'rb' ))
    for metric in ep_metrics.keys():
        logs[ metric ].append( ep_metrics[ metric ] )
    logs[ 'epochs' ].append( epoch )
    colours = [ 'r', 'orange', 'b', 'purple' ]

    fig, ax = plt.subplots()
    xs = np.arange(epoch)
    ax.plot( logs['epochs'], logs['acc'], colours.pop(0), label = 'acc' )
    ax.plot( logs['epochs'], logs['val_acc'], colours.pop(0), label = 'val_acc' )
    ax.set_xlabel( 'Epoch' )
    ax.set_ylim([ 0, 1.3 ])
    ax.set_ylabel( 'Accuracy' )
    ax.legend( loc = 2 )

    ax = ax.twinx()
    ax.plot( logs['epochs'], logs['loss'], colours.pop(0), label = 'loss' )
    ax.plot( logs['epochs'], logs['val_loss'], colours.pop(0), label = 'val_loss' )
    ax.set_ylabel( 'Loss' )
    ax.set_ylim([ 0, 1.3*max(logs['loss'] + logs['val_loss']) ])
    ax.legend( loc = 1 )
    t1 = logs[ 'time_at_last_epoch' ]
    t2 = time()
    dt = t2 - t1
    logs['time_at_last_epoch'] = t2
    logs[ 'runtimes' ].append(dt)
    title_str = 'Training History\n'
    title_str += 'Total Time = %.2fs\n'
    title_str += 'Time of Last Epoch = %.0fs\n'
    title_str += 'Accuracy of Last Epoch = %.4f\n'
    title_str += 'Loss of Last Epoch = %.4f\n'
    title_str += 'Validation Accuracy of Last Epoch = %.4f\n'
    title_str += 'Validation Loss of Last Epoch = %.4f'
    title_params = ( sum(logs['runtimes']), dt,
                     ep_metrics['acc'], ep_metrics['loss'],
                     ep_metrics['val_acc'], ep_metrics['val_loss'] )
    ax.set_title( title_str % title_params )
    plt.show()
    clear_output( wait = True )
    pickle.dump( logs, open( LOGS_PATH, 'wb' ) )

def trainModel(mainPath, trainPath, valPath, EPOCHS = 500, INIT_LR = 1e-3, BS = 32, pxsize = 50, modelName='cae-model.model', plotName='model_perf.png', verboseInfo=0):
    print("[INFO] loading images...")
    trainData = []
    trainLabels = []
    valData = []
    valLabels = []
    trainPathAux = mainPath / trainPath
    valPathAux = mainPath / valPath
    # grab the image paths and randomly shuffle them
    trainImgPaths = sorted(list(paths.list_images(trainPathAux)))
    valImgPaths = sorted(list(paths.list_images(valPathAux)))
    random.seed(1)
    random.shuffle(trainImgPaths)
    random.shuffle(valImgPaths)
    # loop over the train images
    for imagePath in trainImgPaths:
        matplotlib.use("Agg")
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        plt.imshow(image)
        plt.close()
        image = cv2.resize(image, (pxsize, pxsize)) # 28,28
        image = img_to_array(image)
        trainData.append(image)

        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        trainLabels.append(label)
    # loop over the validation images
    for imagePath in valImgPaths:
        matplotlib.use("Agg")
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        plt.imshow(image)
        plt.close()
        image = cv2.resize(image, (pxsize, pxsize)) # 28,28
        image = img_to_array(image)
        valData.append(image)
        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        valLabels.append(label)
    # scale the raw pixel intensities to the range [0, 1]
    trainData = np.array(trainData, dtype="float") / 255.0
    trainLabels = np.array(trainLabels)
    valData = np.array(valData, dtype="float") / 255.0
    valLabels = np.array(valLabels)
    # convert the labels from integers to vectors
    trainY = to_categorical(trainLabels, num_classes=94)
    valY = to_categorical(valLabels, num_classes=94)
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")
    # initialize the model
    print("[INFO] compiling model...")
    model = LeNet.build(width=pxsize, height=pxsize, depth=3, classes=94)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])
    # train the network
    print("[INFO] training network...")
    matplotlib.use("Qt4Agg")
    ###
    logs = pickle.load( open( LOGS_PATH, 'rb' ) )
    logs[ 'time_at_last_epoch' ] = time()
    pickle.dump( logs, open( LOGS_PATH, 'wb' ) )
    plot_metrics_callback = LambdaCallback( on_epoch_end = plot_metrics )
    initial_epoch = 0 if logs['epochs'] == [] else logs['epochs'][-1]
    train_generator.reset()
    ###
    H = model.fit_generator(aug.flow(trainData, trainY, batch_size=BS),
        validation_data=(valData, valY), steps_per_epoch=len(trainData) // BS,
        epochs=EPOCHS, verbose=verboseInfo, shuffle=True, max_queue_size=10, callbacks=[plot_metrics_callback])
    # save the model to disk
    print("[INFO] serializing network...")
    model.save(modelName)
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotName)

def testModel(testPath, pxsize = 50, modelName='cae-model.model'):
    testPath = Path.cwd() / testPath
    listIm = os.listdir(testPath)
    if '.DS_Store' in listIm:
      listIm.remove('.DS_Store')# if .DS_Store exists
    # define variables
    faceAges = []
    predictedList = []
    score = 0;
    range10 = 0;
    range5 = 0;
    outOf = 0;
    idx = 0;
    idx2 = 0;
    layerNames = []
    auxLayerNames = []
    actvList = []
    sumList = []
    ##
    conv2d_1List = []
    activation_1List = []
    max_pooling2d_1List = []
    conv2d_2List = []
    activation_2List = []
    max_pooling2d_2List = []
    ##
    imNameList = []
    activationsList = []
    layerNamesList = []
    for imName in listIm:
        idx2 = 0;
        print(round((100*listIm.index(imName))/len(listIm),2),"%...",end="\r")
        auxName = imName
        imName = testPath / imName
        # load the image
        image = cv2.imread(str(imName))
        orig = image.copy()
        # pre-process the image for classification
        image = cv2.resize(image, (pxsize, pxsize)) # 28,28
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        # load the trained convolutional neural network
        model = load_model(modelName)
        # Save model img to file
        plot_model(model, to_file= str(modelName)[0:-6]+'.png')
        # classify the input image
        # results = model.predict(image)[0]
        # label = np.argmax(results)
        # predictedList.append(label)
        for layer in model.layers[:11]:
            layer_outputs = layer.output
            if idx < 11:
              layerNames.append(layer.name)
              idx += 1
        for layer in model.layers[:11]: # to run make in [:11]
            layer_outputs = layer.output
            activation_model = Model(inputs=model.input, outputs=layer_outputs)
            # Creates a model that will return these outputs, given the model input
            activations = activation_model.predict(image)
            direc = 'intmLayerPlotsHappy/'+layerNames[idx2]
            layer_activation = activations[0]
            if len(layer_activation.shape) == 3:
                if not os.path.exists(direc):
                    os.makedirs(direc)
                plt.matshow(layer_activation[:, :, 4])
                plt.colorbar()
                plt.savefig(direc+'/'+auxName+'_Intm.png')
                plt.close()
            else:
                imNameList.append(imName)
                layerNamesList.append(layerNames[idx2])
                activationsList.append(layer_activation)
                direc = 'intmLayerHeatmapsHappy/'+layerNames[idx2]
                if not os.path.exists(direc):
                    os.makedirs(direc)
                x = np.linspace(1,len(layer_activation),len(layer_activation))
                xloc = np.linspace(0,len(layer_activation),50)
                y = layer_activation[:]
                # fig, (ax,ax2) = plt.subplots(nrows=2, sharex=True)
                extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
                im1 = plt.imshow(y[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
                # ax.set_xticks(xloc)
                # ax.set_yticks([])
                # ax.set_xlim(extent[0], extent[1])
                # ax2.plot(x,y)
                # fig.colorbar(im1, ax=ax, orientation='vertical')
                plt.colorbar(im1, orientation='horizontal')
                plt.tight_layout()
                plt.savefig(direc+'/'+auxName+'_Map.png')
                plt.close()

            idx2+=1
