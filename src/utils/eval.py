import codecs
from sklearn import metrics
import numpy as np
import os

def dense(y):
	label_y = []
	for i in range(len(y)):
		for j in range(len(y[i])):
			label_y.append(y[i][j])

	return label_y
def get_accuracy(y, y_pre):
	sambles = len(y)
	count = 0.0
	for i in range(sambles):
		y_true = 0
		all_y = 0
		for j in range(len(y[i])):
			if y[i][j] > 0 and y_pre[i][j] > 0:
				y_true += 1
			if y[i][j] > 0 or y_pre[i][j] > 0:
				all_y += 1
		if all_y <= 0:
			all_y = 1

		count += float(y_true) / float(all_y)
	acc = float(count) / float(sambles)
	acc=round(acc,4)
	return acc

def get_metrics(y, y_pre):
	"""

	:param y:1071*6
	:param y_pre: 1071*6
	:return:
	"""
	y = y.cpu().detach().numpy()
	y_pre = y_pre.cpu().detach().numpy()
	test_labels = dense(y)
	test_pred = dense(y_pre)

	y=np.array(y)
	y_pre=np.array(y_pre)
	macro_f1 = metrics.f1_score(y, y_pre, average='macro')
	macro_precision = metrics.precision_score(y, y_pre, average='macro')
	macro_recall = metrics.recall_score(y, y_pre, average='macro')
	acc = get_accuracy(y, y_pre)

	micro_f1 = metrics.f1_score(y, y_pre, average='micro')
	micro_precision = metrics.precision_score(y, y_pre, average='micro')
	micro_recall = metrics.recall_score(y, y_pre, average='micro')
	return micro_f1, micro_precision, micro_recall, acc