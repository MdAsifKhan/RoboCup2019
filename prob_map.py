from arguments import opt
import os
import pdb
import xmltodict as xd
import numpy as np
from scipy.stats import multivariate_normal
import torch
import h5py
from tqdm import tqdm
import math

class ProbMap:
	'''
	Class to create probability maps from bounding box and save it as h5py file on disk 
	'''
	def __init__(self, dataroot):
		self.dataroot = dataroot
		self.prob_maps = []
		self.image_name = []
		self.centers = []
		self.min_radius = 0

	def create_prob_map(self):
		for filename in tqdm(os.listdir(self.dataroot)):
			name = filename[:-4]
			if filename.endswith('.xml'):
				with open(self.dataroot + '/' + filename) as f:
					tree = xd.parse(f.read())

				if type(tree['annotation']['object']) is not list:
					tree['annotation']['object'] = [tree['annotation']['object']]
			
				prob_map_ = np.zeros([120, 160], dtype='float32')
				center = np.array([-1, -1])
				min_radius = 1200
				for object_ in tree['annotation']['object']:
					if object_['name']=='ball':
						bndbox = object_['bndbox1']
						xmin, ymin = int(bndbox['xmin'])/4, int(bndbox['ymin'])/4
						xmax, ymax = int(bndbox['xmax'])/4, int(bndbox['ymax'])/4
						center = np.array([(ymax+ymin)/2, (xmax+xmin)/2])
						radius = min((xmax-xmin)/2, (ymax-ymin)/2)
						min_radius = min(min_radius, radius)
						prob_map1_ = self.prob_map(prob_map_, xmin, ymin, xmax, ymax, center, radius)						
					try:
						bndbox1 = object_['bndbox2']
						xmin1, ymin1 = int(bndbox1['xmin'])/4, int(bndbox1['ymin'])/4
						xmax1, ymax1 = int(bndbox1['xmax'])/4, int(bndbox1['ymax'])/4
						center1 = np.array([(ymax+ymin)/2, (xmax+xmin)/2])
						radius1 = min((xmax-xmin)/2, (ymax-ymin)/2)
						min_radius1 = min(min_radius, radius1)
						prob_map2_ = self.prob_map(prob_map_, xmin1, ymin1, xmax1, ymax1, center1, radius1)
					except:
						continue
				try:			
					join_prob_map_ = prob_map1_ + prob_map2_
				except:
					join_prob_map_ = prob_map1_
					center1 = np.asarray([-1, -1])
				centers = np.asarray([center, center1])
				self.prob_maps.append(join_prob_map_*100)
				self.image_name.append(name)
				self.centers.append(centers)
				self.min_radius = min(min_radius, self.min_radius)

	def prob_map(self, prob_map_, xmin, ymin, xmax, ymax, center, radius=4):
		for x in range(int(ymin), min(math.ceil(ymax), prob_map_.shape[0])):
			for y in range(int(xmin), min(math.ceil(xmax), prob_map_.shape[1])):
				prob_map_[x, y] = multivariate_normal.pdf([x, y], center, [radius, radius])
		return prob_map_

	def save_prob_map(self, data_file):
		prob_maps = np.asarray(self.prob_maps, dtype='float32')
		self.centers = np.asarray(self.centers, dtype='float32')
		with h5py.File(self.dataroot + '/' + data_file, 'w') as hf:
			hf.create_dataset('prob_maps', data = prob_maps)
			self.image_name = [n.encode('ascii', 'ignore') for n in self.image_name]
			hf.create_dataset('filenames', data = self.image_name)
			hf.create_dataset('centers', data = self.centers)
			hf.create_dataset('min_radius', data=self.min_radius)

if __name__=='__main__':

	prob_train = ProbMap(opt.data_root+'/train_cnn')
	prob_train.create_prob_map()
	prob_train.save_prob_map(data_file='train_maps')

	prob_test = ProbMap(opt.data_root+'/test_cnn')
	prob_test.create_prob_map()
	prob_test.save_prob_map(data_file='test_maps')
