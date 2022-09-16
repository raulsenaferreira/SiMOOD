import literature_safety_monitors as sm
import numpy as np
from numpy.random import seed

import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

import pandas as pd 
from matplotlib import pyplot as plt
import cv2
import os
from xml.etree import ElementTree





def plot_accuracy_loss(history):
	"""
		Plot the accuracy and the loss during the training of the nn.
	"""
	fig = plt.figure(figsize=(10,5))

	# Plot accuracy
	plt.subplot(221)
	plt.plot(history.history['accuracy'],'bo--', label = "acc")
	plt.plot(history.history['val_accuracy'], 'ro--', label = "val_acc")
	plt.title("train_acc vs val_acc")
	plt.ylabel("accuracy")
	plt.xlabel("epochs")
	plt.legend()

	# Plot loss function
	plt.subplot(222)
	plt.plot(history.history['loss'],'bo--', label = "loss")
	plt.plot(history.history['val_loss'], 'ro--', label = "val_loss")
	plt.title("train_loss vs val_loss")
	plt.ylabel("loss")
	plt.xlabel("epochs")

	plt.legend()
	plt.show()


def load_data(size):
	'''
	Citation
	@inproceedings{inproceedings,
	author = {N J Karthika, Chandran Saravanan},
	booktitle = {International Conference on Electronic Systems and Intelligent Computing (ESIC 2020)}
	year = {2020},
	month = {3),
	title = {Addressing False Positives in Pedestrian Detection}
	doi = {https://doi.org/10.1007/978-981-15-7031-5_103}
	}
	'''
	datasets = ['pedestrian_dataset/Train/Train', 'pedestrian_dataset/Test/Test', 'pedestrian_dataset/Val/Val']
	output = []

	for dataset in datasets:
		imags = []
		labels = []
		directoryA = dataset +"/Annotations"
		directoryIMG = dataset +"/JPEGImages/"
		file = os.listdir(directoryA)
		img = os.listdir(directoryIMG)
		file.sort()
		img.sort()

		i = 0
		for xml in file:

			xmlf = os.path.join(directoryA,xml)
			dom = ElementTree.parse(xmlf)
			vb = dom.findall('object')
			label = vb[0].find('name').text
			labels.append(class_names_label[label])

			img_path = directoryIMG + img[i]
			curr_img = cv2.imread(img_path)
			curr_img = cv2.resize(curr_img, size)
			imags.append(curr_img)
			i +=1
		
		imags = np.array(imags, dtype='float32')
		imags = imags / 255
		
		labels = np.array(labels, dtype='int32')

		output.append((imags, labels))
	return output


def model_train(size, train_images, train_labels, test_images, test_labels):

	model = models.Sequential()
	model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(size[0], size[1], 3)))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(32, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Flatten())
	model.add(layers.Dense(128, activation='relu'))
	model.add(layers.Dense(2))
	model.summary()

	model.compile(optimizer='adam',
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=['accuracy'])
	es = EarlyStopping(monitor='accuracy', mode='max', verbose=1, patience=7)
	filepath = "modelPedestrianDetection.h5"
	ckpt = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
	rlp = ReduceLROnPlateau(monitor='accuracy', patience=3, verbose=1)

	history = model.fit(train_images, train_labels, epochs=60,
		validation_data=(test_images, test_labels),
		callbacks=[es,ckpt,rlp])

	return history, model


def pedestrian_model_training():
	seed(1)
	keras = tf.keras

	class_names = ['person','person-like']
	class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

	n_classes = 2
	size = (120,120)

	(train_images, train_labels),(test_images, test_labels),(val_images, val_labels) = load_data(size)

	print('train_images.shape',train_images.shape)

	plt.figure(figsize=(20,20))
	for n , i in enumerate(list(np.random.randint(0,len(train_images),36))) : 
		plt.subplot(6,6,n+1)
		plt.imshow(train_images[i])  
		plt.title(class_names[train_labels[i]])
		plt.axis('off')

	history, model = model_train(size, train_images, train_labels, test_images, test_labels)

	plot_accuracy_loss(history)

	preds = model.predict(val_images) 

	plt.figure(figsize=(20,20))
	for n , i in enumerate(list(np.random.randint(0,len(val_images),36))) : 
		plt.subplot(6,6,n+1)
		plt.imshow(val_images[i])	
		plt.axis('off')
		x =np.argmax(preds[i]) # takes the maximum of of the 6 probabilites. 
		plt.title((class_names[x]))

	result = []
	for i in range(len(preds)):
		result.append(np.argmax(preds[i]))

	tn, fp, fn, tp = confusion_matrix(val_labels,result).ravel()
	print('tn, fp, fn, tp', tn, fp, fn, tp)


if __name__ == '__main__':
	#pedestrian_model_training()

	