from keras.models import load_model
import argparse
import pickle
import cv2
import os
import random

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image we are going to classify")
ap.add_argument("-m", "--model", required=True,
	help="path to trained Keras model")
ap.add_argument("-l", "--labelBin", required=True,
	help="path to label binarizer")
ap.add_argument("-w", "--width", type=int, default=96,
	help="target spatial dimension width")
ap.add_argument("-e", "--height", type=int, default=96,
	help="target spatial dimension height")

args = vars(ap.parse_args())

print("[INFO] loading model...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelBin"], 'rb').read())

while True:
	imagePath = random.choice(os.listdir(args['image']))
	image = cv2.imread(args['image']+os.sep+imagePath)
	output = image.copy()
	image = cv2.resize(image, (args['width'], args['height']))

	# normalize pixel
	image = image.astype('float')/255.0


	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

	# prediciton on image
	preds = model.predict(image) # array([[5.4622066e-01, 4.5377851e-01, 7.7963534e-07]], dtype=float32) 2D array

	# find the class label index with highest probability
	i = preds.argmax(axis=1)[0]
	label = lb.classes_[i]

	# display result
	trueLabel = imagePath.split('_')[1].split('.')[0]
	correctPrediction = "Correct" if label == trueLabel else "Wrong"
	correctPredictionColor = (0,255,0) if correctPrediction =="Correct" else (0,0,255)

	text_answer = "Answer: {}".format(trueLabel)
	text_prediction = "Predicted: {}: {:.2f}%".format(label, preds[0][i] *100)
	text_ifCorrect = "{}".format(correctPrediction)


	cv2.putText(output, text_ifCorrect, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, correctPredictionColor,2)
	cv2.putText(output, text_prediction, (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255),2)
	cv2.putText(output, text_answer, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0),2)

	cv2.imshow("Image", output)
	cv2.waitKey(0)
	
	if cv2.waitKey(0) == ord("q"):
		print("[INFO] Exit Program...")
		break












