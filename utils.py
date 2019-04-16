import numpy as np
import pdb
from arguments import opt
import torch
import cv2
from scipy import ndimage


def performance_metric(TP, FP, FN, TN):
	
	accuracy = (TP+TN)/(TP+TN+FP+FN)
	if FP==0 and TP==0:
		FDR = 1.0
	else:
		FDR = FP/(FP + TP)
	if FN==0 and TP==0:
		RC = 0.0
	else:
		RC = TP/(TP + FN)
	return FDR, RC, accuracy


def post_processing(maps, threshold):

	processed_maps, predicted_centers, maps_area = [], [], []
	for map_ in maps:
		binary_map = (map_>0.1).astype(np.uint8)
		contours = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
		img = np.zeros(map_.shape, np.uint8)
		if len(contours)>0:
			contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
			area, biggest_contour = max(contour_sizes, key=lambda x: x[0])
			contour_sizes.remove((area, biggest_contour))
			area2, 2nd_contour = max(contour_sizes, key=lambda x: x[0])

			# Biggest Contour
			cv2.drawContours(img, [biggest_contour], -1, (1), cv2.FILLED)
			apply_map = map_ * img
			cX, cY = ndimage.measurements.center_of_mass(apply_map)

			# 2nd Biggest Contour
			img = np.zeros(map_.shape, np.uint8)
			cv2.drawContours(img, [2nd_contour], -1, (1), cv2.FILLED)
			apply_map1 = map_ * img
			cX1, cY1 = ndimage.measurements.center_of_mass(apply_map)
			
			two_centers = [(cX, cY), (cX1, cY1)]
			two_areas = [[area], [area2]]
			maps_detect = [apply_map, apply_map1]
			predicted_centers.append(two_centers)
			maps_area.append(two_areas)

			processed_maps.append(maps_detect)
		else:
			apply_map = map_ * img
			processed_maps.append(apply_map)
			maps_area.append([[0], [0]])
			predicted_centers.append([(-1, -1),(-1, -1)])
	return processed_maps, predicted_centers, maps_area


def tp_fp_tn_fn_alt(actual_centers, predicted_centers, maps_area, min_radius):
	
	minm_area = min_radius**2
	TP, FP, TN, FN = 0, 0, 0, 0
	for areas, act_centers, pred_centers in zip(maps_area, actual_centers, predicted_centers):
		for area, (a_x, a_y), (p_x, p_y) in zip(areas, act_centers, pred_centers):
			if a_x==-1 and a_y==-1 and (area<minm_area or (p_x==-1 and p_y==-1)):
				TN += 1
			elif (a_x>=0 and a_y>=0) and area<minm_area:
				FN += 1
			elif (a_x>=0 or a_y>=0) and area>=minm_area and within_radius(a_x, a_y, p_x, p_y, min_radius):
				TP += 1
			else:
				FP +=1
	return TP, FP, TN, FN
