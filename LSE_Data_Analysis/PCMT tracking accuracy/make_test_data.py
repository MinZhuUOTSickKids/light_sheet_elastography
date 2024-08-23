#!/usr/bin/env python3

import os
import numpy as np
from matplotlib import pyplot as plt
import secrets
from PIL import Image
from pathlib import Path
import shutil
from scipy.ndimage import shift

file_size = 512
time_size = 128
dot_number = 24
dot_size = 0.8**2
buffer = 48
pos = file_size/2
Y,X = np.meshgrid(np.arange(file_size), np.arange(file_size))
point = np.floor(256*np.exp(-((X-pos)**2+(Y-pos)**2)/dot_size)).astype(int)

rng = np.random.default_rng(secrets.randbits(128))
data_array = np.zeros((time_size, 1, 1, file_size, file_size))

amplitude = buffer/8 + rng.normal(0.0, 0.1, size=dot_number)
frequency = np.pi*np.euler_gamma
initial_pos = rng.random(size=(dot_number, 2))*(file_size-2*buffer) + buffer
phase = initial_pos[:,0]/1000
noise = rng.normal(0.0, 0.2, size=(time_size, dot_number, 2))
position = np.zeros((time_size, dot_number, 2))


output_array = np.zeros((time_size*dot_number,5), dtype = float)
test_data_path = Path('./test_data')
Path.mkdir(test_data_path, exist_ok = True)
for child in test_data_path.iterdir():
	child.unlink()

counter = 1
for time_point in range(time_size):
	position[time_point,:,:] = initial_pos + noise[time_point,:,:]
	position[time_point,:,1] += amplitude*np.sin(frequency*time_point + phase)
	for dot_index in range(dot_number):
		x, y = position[time_point, dot_index, :]
		data_array[time_point,0,0,:,:] += shift(point, (x-pos, y-pos))
		output_array[time_point*dot_number+dot_index] = [
				position[time_point,dot_index,1],
				position[time_point,dot_index,0],
				time_point+1, dot_index+1, counter]
		counter += 1
	image = Image.fromarray(data_array[time_point,0,0,:,:])
	image.save(test_data_path/f'test_T{time_point+1:03d}_Z001.tiff')

shutil.make_archive(Path('./test_data'), 'zip', test_data_path)
for child in test_data_path.iterdir():
	child.unlink()
test_data_path.rmdir()

data_format = '%.18e', '%.18e', '%1d', '%1d', '%1d'
np.savetxt(Path('./test_data_positions.csv'),
			output_array,  delimiter = ',',
			fmt = data_format,
			header = 'X,Y,Time,TrackID,ID')


fig = plt.figure(2, figsize = (16,8))
ax1 = plt.subplot(121)
ax1.imshow(data_array[0,0,0,:,:])
ax1.set_xlim([0,file_size])
ax1.set_ylim([0,file_size])
ax2 = plt.subplot(122)
ax2.imshow(data_array[4,0,0,:,:])
ax2.set_xlim([0,file_size])
ax2.set_ylim([0,file_size])
#plt.plot(position[0,:,1], position[0,:,0], '.k')
#plt.plot(position[2,:,1], position[2,:,0], '.r')
#plt.plot(position[4,:,1], position[4,:,0], '.b')
plt.show()


