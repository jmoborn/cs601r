import numpy as np
import cv2

import train_svm as train

features_filename = 'orb_training_features_large.txt'
data = []
with open(features_filename, 'r') as features_file:
	for line in features_file:
		features = [float(x) for x in line.strip().split()]
		data.append(features)

# we only need butterfly data
data = data[:418]
labels = train.get_classes_train()

data_dict = {}
for class_idx in xrange(10):
	data_dict[class_idx] = []
for idx, data_array in enumerate(data):
	data_dict[labels[idx]-1].append(data_array)

classes = 10
samples = 20
dimx = classes*samples
dimy = len(data[0])

image = np.zeros((dimy, dimx, 1), np.uint8)
img_idx = 0
for pixel in np.nditer(image, op_flags=['readwrite']):
	img_x = img_idx % dimx
	img_y = img_idx / dimx
	class_idx = img_x / samples
	class_sample_idx = img_x % classes
	pixel[...] = data_dict[class_idx][class_sample_idx][img_y]*28
	img_idx += 1

color = cv2.applyColorMap(image, cv2.COLORMAP_JET)
cv2.imwrite('visualize_histogram.ppm', color)