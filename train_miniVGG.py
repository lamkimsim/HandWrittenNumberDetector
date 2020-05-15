from keras.datasets.mnist import load_data
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import to_categorical
from CNN_Model.miniVGG import miniVGG

import matplotlib.pyplot as plt
import numpy as np 
import imutils 
import cv2
import os
import argparse

args = argparse.ArgumentParser()
args.add_argument('-p', '--plot', default='plot.png')
args = vars(args.parse_args())

(trainX, trainY), (testX, testY) = load_data()
classes = len(list(set(trainY)))
trainY = to_categorical(trainY, num_classes=classes)
testY = to_categorical(testY, num_classes=classes)

trainX = trainX.reshape(trainX.shape[0], 28, 28, 1)
testX = testX.reshape(testX.shape[0], 28, 28, 1)

trainX = trainX.astype("float32")
testX = testX.astype("float32")

trainX = trainX/255.0
testX = testX/255.0

# data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,height_shift_range=0.1, 
	                     shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode="nearest")


model = miniVGG.fit(width=28, height=28, depth=3, classes=classes)

LR = 0.01
EPOCHS = 75
BATCH_SIZE = 32

print("[INFO] training model...")
opt = SGD(lr=LR, decay=LR/EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
	                    validation_data=(testX, testY), steps_per_epoch=len(trainX) // BATCH_SIZE,
	                    epochs=EPOCHS)

# save model
model.save("miniVGG.model")

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (SmallVGGNet)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])