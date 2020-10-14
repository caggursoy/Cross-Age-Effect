# USAGE
# python train_network.py --trainset /images/train --valset /images/val --model cae-model.model
# import PlaidML first
import plaidml.keras
plaidml.keras.install_backend()
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from keras.utils import to_categorical
from keras.utils import plot_model
from pyimagesearch.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
from pathlib import Path
# import fileSort

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--trainset", required=True,
	help="path to training dataset")
ap.add_argument("-v", "--valset", required=True,
	help="path to validation dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 500
INIT_LR = 1e-3  #1e-3
BS = 32

# model px size
pxsize = 50

# initialize the data and labels
print("[INFO] loading images...")
trainData = []
trainLabels = []
valData = []
valLabels = []

mainPath = 'D:\Repos\Cross-Age-Effect\caeMLGPU'
trainPath = mainPath + "/" + args["trainset"]
valPath = mainPath + "/" + args["valset"]
# grab the image paths and randomly shuffle them
trainImgPaths = sorted(list(paths.list_images(trainPath)))
valImgPaths = sorted(list(paths.list_images(valPath)))
random.seed(1)
random.shuffle(trainImgPaths)
random.shuffle(valImgPaths)

# loop over the train images
for imagePath in trainImgPaths:
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

# # partition the data into training and testing splits using 60% of
# # the data for training and the remaining 20% for testing and 20% for validation
# (trainX, testX, trainY, testY) = train_test_split(data,
# 	labels, test_size=0.2, random_state=1)

# # validation split
# (trainX, valX, trainY, valY) = train_test_split(trainX,
# 	trainY, test_size=0.25, random_state=1)

# convert the labels from integers to vectors
trainY = to_categorical(trainLabels, num_classes=94)
# testY = to_categorical(testY, num_classes=94)
valY = to_categorical(valLabels, num_classes=94)

print(max(trainY), max(valY))

# print("Training size: ",str(int(100*trainX.size/data.size)),"%")
# print("Test size: ",str(int(100*testX.size/data.size)),"%")
# print("Validation size: ",str(int(100*valX.size/data.size)),"%")

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
H = model.fit_generator(aug.flow(trainData, trainY, batch_size=BS),
	validation_data=(valData, valY), steps_per_epoch=len(trainData) // BS,
	epochs=EPOCHS, verbose=1, shuffle=True, max_queue_size=10)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

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
plt.savefig(args["plot"])
