import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sklearn.cluster import KMeans
from matplotlib.path import Path
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely import geometry, affinity
from keras.models import Model



### A set of monitors from the literature

def ALOOC():
	pass


def ODIN():
	pass


class OOB_monitor():

	def __init__(self, num_clusters=3):
		self.num_clusters = num_clusters
		self.arr_abstractions_by_class = []

	
	def get_activ_func(self, model, image, layerIndex):
		inter_output_model = Model(inputs = model.input, outputs = model.get_layer(index=layerIndex).output)
		return inter_output_model.predict(image)


	def do_abstract(self, weights_neuron):
		weights_neuron = np.asarray(weights_neuron)

		#print("reducing data...", weights_neuron.shape)

		#doing a projection by taking just the first and the last dimension of data
		weights_neuron = weights_neuron[:,[0,-1]]

		#print("making boxes by cluster...", weights_neuron.shape)
		
		x1 = np.amin(weights_neuron[:,0])
		x2 = np.amax(weights_neuron[:,0])
		y1 = np.amin(weights_neuron[:,1])
		y2 = np.amax(weights_neuron[:,1])

		rectangle = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

		return Polygon(rectangle)


	def do_abstract_by_cluster(self, dataByCluster):
		arr_polygon = []

		for cluster, weights_neuron in dataByCluster.items():
			polygon = self.do_abstract(weights_neuron)
			arr_polygon.append(polygon)
			#abstract_box = geometry.box(x1, y1, x2, y2)
			#arr_polygon.append(abstract_box)

		return arr_polygon
		

	def make_abstraction(self, activations_by_class):
		
		os.makedirs(self.path_to_save, exist_ok=True)

		abstractions_by_class = None

		for cls, activations in activations_by_class.items():
		
			if self.num_clusters > 0:
				dataByCluster={}
				clusters = KMeans(n_clusters=self.num_clusters).fit_predict(activations)

				for c, d in zip(clusters, activations):
					try:
						dataByCluster[c].append(d)
					except:
						dataByCluster.update({c:[d]})

				abstractions_by_class = self.do_abstract_by_cluster(dataByCluster)

			else:
				abstractions_by_class = self.do_abstract(activations)

			arr_abstractions_by_class.append(abstractions_by_class)

		return arr_abstractions_by_class


	def build_monitor(self, model, X, y):
		activations_by_class = {}

		for img, lab in zip(X, y):

			img = np.asarray([img])
			yPred = np.argmax(model.predict(img))
			
			if yPred == lab:
				activation = self.get_activ_func(model, img, layerIndex=-2)[0]
				
				try:
					activations_by_class[yPred].append(activation)
				except:
					activations_by_class.update({yPred: [activation]})
			
		self.arr_abstractions_by_class = self.make_abstraction(activations_by_class)