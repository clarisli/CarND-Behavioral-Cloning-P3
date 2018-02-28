import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from random import shuffle
from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Dropout, Reshape, SpatialDropout2D
from keras.layers import Cropping2D, MaxPooling2D, Conv2D

# Step 0: Load The Data

samples = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader,None)
	for line in reader:
		samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Step 1: Dataset Summary & Exploration

def sample_img(source_path):
	filename = source_path.split('/')[-1]
	path = './data/IMG/' + filename
	image = cv2.imread(path)
	return image


def balance_distribution(samples, num_bins=25):
	bin_edges = np.arange(-1,1,2/num_bins)
	avg_bin_count = int(len(samples)/num_bins)
	for idx, value in enumerate(bin_edges[:-1]):
		bin_samples = [s for s in samples if float(s[3]) >= bin_edges[idx] and float(s[3]) < bin_edges[idx+1]]
		bin_count = len(bin_samples)
		if bin_count > 0:
			if bin_count < avg_bin_count:
				multiplier = int(avg_bin_count/bin_count)
				remainder = int(avg_bin_count%bin_count)       
				samples.extend(bin_samples*multiplier)
				samples.extend(bin_samples[:remainder])
			else:
				unwanted = bin_samples[:int((bin_count-avg_bin_count)/3)]                
				samples = [s for s in samples if s not in unwanted]
	return samples
train_samples = balance_distribution(train_samples)

# Data Augmentation
# mode: random, center, left, right
def camera_data(sample, mode='center'):
	correction = 0.2
	center_image = sample_img(sample[0])
	center_angle = float(sample[3])
	left_image = sample_img(sample[1])
	left_angle = center_angle + correction
	right_image = sample_img(sample[2])
	right_angle = center_angle - correction
	if mode == 'random':
		if np.random.rand() > 0.9:
			if np.random.rand() > 0.5:
				return left_image, left_angle
			else:
				return right_image, right_angle
	elif mode == 'left':
		return left_image, left_angle
	elif mode == 'right':
		return right_image, right_angle
	
	return center_image, center_angle


def brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2BGR)
    return image1

def translate(image, steering):
    trans = 100*np.random.uniform()-50
    M = np.float32([[1,0,trans],[0,1,trans]])
    image = cv2.warpAffine(image, M, image.shape[::-1][1:])
    steering += trans*0.002
    return image, steering

def flip(img, steering):
	image_flipped = np.fliplr(img)
	steering_flipped = -steering
	return image_flipped, steering_flipped

def augment(image, angle):
	if np.random.rand() > .5:
		image, angle = flip(image, angle)
	if np.random.rand() > .5:
		image = brightness(image)
	if np.random.rand() > .8:
		image, angle = translate(image, angle)
	return image, angle


def generator(samples, batch_size=32, augment_data=False):
	num_samples = len(samples)
	while 1:
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:
				if augment_data:
					image, angle = camera_data(batch_sample, mode='random')
					image, angle = augment(image, angle)
				else:
					image, angle = camera_data(batch_sample, mode='center')
				images.append(image)
				angles.append(angle)

				if (len(images) >= batch_size):
					X_train = np.array(images)
					y_train = np.array(angles)
					yield sklearn.utils.shuffle(X_train, y_train)

	
train_generator = generator(train_samples, augment_data=True)
validation_generator = generator(validation_samples)	

# Models

input_shape = (160,320,3)

def rgb2hsv(x):
	import tensorflow as tf
	channels = tf.unstack (x, axis=-1)
	image = tf.stack([channels[2], channels[1], channels[0]], axis=-1)
	hsv = tf.image.rgb_to_hsv(x)
	return hsv[:,:,:,1:2]


def nvidia_model():
	model = Sequential()
	model.add(Lambda(lambda x: x / 255.0, input_shape=(160,320,3)))
	model.add(Lambda(rgb2hsv))
	model.add(Cropping2D(cropping=((50,20),(0,0))))
	model.add(Conv2D(24,5,5, subsample=(2,2), activation='relu'))
	model.add(Conv2D(36,5,5, subsample=(2,2), activation='relu'))
	model.add(Conv2D(48,5,5, subsample=(2,2), activation='relu'))
	model.add(Conv2D(64,3,3, activation='relu'))
	model.add(Conv2D(64,3,3, activation='relu'))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	return model


model = nvidia_model()
model.summary()
model.compile(optimizer='adam',loss='mse')
history_object = model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)
model.save('model.h5')


