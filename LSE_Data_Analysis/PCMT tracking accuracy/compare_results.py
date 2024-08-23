#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

# header = 'X,Y,Time,TrackID,ID'

position_data = np.loadtxt(Path('./test_data_positions.csv'),
						delimiter = ',', comments = '#')
time_points = np.sort(np.unique(position_data[:,2]))
cell_ids = position_data[position_data[:,2] == time_points[0], 3]
positions = np.zeros((len(cell_ids), len(time_points), 2), dtype = float)
for time_index, time_point in enumerate(time_points):
	for cell_index, cell_id in enumerate(cell_ids):
		positions[cell_index,time_index,:] = position_data[
			np.logical_and(position_data[:,2] == time_point,
						   position_data[:,3] == cell_id), 0:2]

result_data = np.loadtxt(Path('./test_data_results.csv'),
						delimiter = ',', comments = '#')
initial_positions = result_data[result_data[:,2] == time_points[0], 0:2]
result_ids = np.zeros_like(cell_ids)
result_shift = np.zeros((len(cell_ids),2), dtype = float)
for index, position in enumerate(initial_positions):
	distances = np.linalg.norm(positions[:,0,:] - position[np.newaxis,:],
									axis=1)
	closest = np.argmin(distances)
	result_ids[index] = cell_ids[closest]
	result_shift[index] = positions[closest,0,:] - position
result_ids = cell_ids[np.argsort(result_ids)]

results = np.zeros((len(cell_ids), len(time_points), 2), dtype = float)
for time_index, time_point in enumerate(time_points):
	for cell_index, cell_id in enumerate(result_ids):
		results[cell_index,time_index,:] = result_data[
			np.logical_and(result_data[:,2] == time_point,
						   result_data[:,3] == cell_id), 0:2] #- \
	#			result_shift[cell_index]

distances = np.linalg.norm(positions-results, axis=-1)
#distances = positions-results
print("absolute error: ", np.mean(distances), "±", np.std(distances))
shifts = (positions - np.roll(positions, 1, axis=1))[:,1:,:]
result_shifts = (results - np.roll(results, 1, axis=1))[:,1:,:]
distances = np.linalg.norm(shifts - result_shifts, axis=-1)
print("point to point shift error: ", np.mean(distances), "±", np.std(distances))
