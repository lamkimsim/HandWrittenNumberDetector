from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K


class LeNet:
	@staticmethod
	def build(numChannels, height, width, classes, activation='relu', weightsPath=None):

		model = Sequential()
		inputShape = (height, width, numChannels)

		if K.image_data_format() == "channels_first":
			inputShape = (numChannels, height, width)


		model.add(Conv2D(20, 5, padding='same', activation=activation, input_shape=inputShape))
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

		model.add(Conv2D(50, 5, padding='same', activation=activation, input_shape=inputShape))
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

		model.add(Flatten())
		model.add(Dense(512, activation=activation))

		# define 2nd FC Layer
		model.add(Dense(classes,activation="softmax"))

		if weightsPath is not None:
			model.load_weights(weightsPath)

		return model




































