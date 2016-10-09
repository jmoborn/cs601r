from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn import preprocessing
import cv2
import numpy as np
import os
import sys

classify = True
randomforest = False
lbp = True

orb_norm = 1.0

def _get_classes(train=True):
	data_dir = 'images/leedsbutterfly/images'
	names = os.listdir(data_dir)
	classes = []
	for name in names:
		filenumber = int(os.path.splitext(name)[0].split('_')[-1])
		if (filenumber % 2 == 0 and not train) or (filenumber % 2 == 1 and train):
			classes.append(int(name.split('_')[0]))
	return classes

def get_classes_train():
	return _get_classes()

def get_classes_test():
	return _get_classes(False)

def train():
	features_filename = 'orb_training_features_large.txt'
	data = []
	with open(features_filename, 'r') as features_file:
		for line in features_file:
			features = [float(x)/orb_norm for x in line.strip().split()]
			data.append(features)

	if lbp:
		with open('lbp_training_features_large.txt', 'r') as features_file:
			for idx, line in enumerate(features_file):
				lbp_features = [float(x) for x in line.strip().split()]
				# print len(lbp_features)
				data[idx].extend(lbp_features)

	labels = []
	if classify:
		data = data[:418]
		labels = get_classes_train()
	else:
		for i in xrange(418):
			labels.append(1)
		for i in xrange(398):
			labels.append(0)

	print len(data)
	print len(labels)
	# data = preprocessing.normalize(data)

	if randomforest:
		classifier = RandomForestClassifier()
		print 'using random forest...'
	else:
		classifier = svm.LinearSVC(C=1.0)
	classifier.fit(data, labels)

	classifier_pickle = 'orb_classifier_large.pkl'
	if os.path.exists(classifier_pickle):
		os.remove(classifier_pickle)
	joblib.dump(classifier, classifier_pickle)

def predict():
	features_filename = 'orb_test_features_large.txt'
	data = []
	with open(features_filename, 'r') as features_file:
		for line in features_file:
			features = [float(x)/orb_norm for x in line.strip().split()]
			data.append(features)

	if lbp:
		with open('lbp_test_features_large.txt', 'r') as features_file:
			for idx, line in enumerate(features_file):
				data[idx].extend([float(x) for x in line.strip().split()])

	positives = 414
	negatives = 434
	labels = []
	if classify:
		# only use butterfly data
		data = data[:positives]
		labels = get_classes_test()
	else:
		for i in xrange(positives):
			labels.append(1)
		for i in xrange(negatives):
			labels.append(0)

	print len(data)
	print len(labels)

	classifier = joblib.load('orb_classifier_large.pkl')

	classes = 10
	confusion_matrix = [None]*classes
	for i in xrange(classes):
		confusion_matrix[i] = [0]*classes

	predictions = []

	threshold = 0.6
	samples = len(data)
	correct = 0
	for i in xrange(samples):
		x = np.array(data[i])
		x_ndarray = x.reshape(1,-1)
		guess = classifier.predict(x_ndarray)

		if guess == labels[i]:
			correct += 1

		if not classify and not randomforest:
			predictions.append((classifier.decision_function(x_ndarray), labels[i]))

		if classify:
			confusion_matrix[guess-1][labels[i]-1] += 1

	print float(correct) / float(samples)

	if not classify and not randomforest:
		# generate precision-recall curves
		predictions.sort()
		dimx = 500
		dimy = 500
		padx = 50
		pady = 50
		pr_img = np.zeros((dimx+padx*2, dimy+pady*2, 1), np.uint8)
		ln_color = (255,255,255)
		ln_width = 1
		cv2.line(pr_img, (padx, pady), (padx+dimx, pady), ln_color, 2)
		cv2.line(pr_img, (padx, pady), (padx, pady+dimy), ln_color, 2)
		TP = float(positives)
		FP = float(negatives)
		FN = 0.0
		prev_pt = None
		for prediction in predictions:
			precision = TP / (TP + FP)
			recall = TP / (TP + FN)
			cur_pt = (int(recall*dimx) + padx, int(precision*dimy) + pady)
			if prev_pt:
				cv2.line(pr_img, prev_pt, cur_pt, ln_color, ln_width)

			if prediction[1] == 1:
				TP -= 1
				FN += 1
			else:
				FP -= 1
			prev_pt = cur_pt
		cv2.imwrite('pr_curve.ppm', pr_img)

	if classify:
		class_totals = [0.0]*classes
		for x in xrange(classes):
			for y in xrange(classes):
				class_totals[x] += confusion_matrix[x][y]
		# print confusion matrix
		dimx = 400
		dimy = 400
		cm_img = np.zeros((dimx, dimy, 1), np.uint8)
		for x in xrange(classes):
			row_total = 0.0
			for y in xrange(classes):
				cm_flt = float(confusion_matrix[x][y])/class_totals[x]
				cm_str = '%.3f' % (cm_flt)
				cm_clr = int(cm_flt*255)
				pt1 = (int(float(x)/classes * dimx), int(float(y)/classes * dimy))
				pt2 = (int(float(x+1)/classes * dimx), int(float(y+1)/classes * dimy))
				cv2.rectangle(cm_img, pt1, pt2, cm_clr, -1)
				txt_clr = 255
				if x-1 == y-1:
					txt_clr = 0
				cv2.putText(cm_img, cm_str, (pt1[0]+1, pt2[1]-2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, txt_clr)
				row_total += cm_flt
		cm_color_img = cv2.applyColorMap(cm_img, cv2.COLORMAP_COOL)
		cv2.imwrite('confusion.ppm', cm_color_img)

if __name__ == '__main__':
	usage = 'USAGE: %s train | predict' % sys.argv[0]
	if len(sys.argv) < 2:
		print usage
	elif sys.argv[1] == 'train':
		train()
	elif sys.argv[1] == 'predict':
		predict()
	else:
		print usage