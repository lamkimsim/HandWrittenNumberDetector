from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras import backend as K


class miniVGG:
	@staticmethod
	def fit(width, height, depth, classes):
		chanDim = -1
		depth = 1

		inputShape = (height, width, depth)

		if K.image_data_format() =="channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		model = Sequential()

		model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=inputShape))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Dropout(0.2))
		model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Dropout(0.2))

		model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Dropout(0.2))
		model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Dropout(0.2))

		model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
		model.add(Dropout(0.2))

		model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Dropout(0.2))
		model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Dropout(0.2))

		model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
		model.add(Dropout(0.2, seed=42))

		model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Dropout(0.2))
		model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Dropout(0.2))

		model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
		model.add(Dropout(0.2, seed=42))

		model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Dropout(0.2))
		model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Dropout(0.2))

		model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
		model.add(Dropout(0.2, seed=42))

		model.add(Flatten())
		model.add(Dense(1024, activation='relu'))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Dropout(0.2))

		# softmax classifier
		model.add(Dense(classes, activation='softmax'))

		return model
