import random
import numpy as np
import pickle
import copy
import sys
import time
import tensorflow as tf 

pstri = "./pickle_jar/"
fs = open(pstri + "tar_data_with_dic.pickle", "rb")
tup_t = pickle.load(fs)
fs.close()
target_arr, target_label,  target_dic = tup_t
dum_arr = target_label.reshape((target_label.shape[0], 1))
clumped_arr = np.concatenate((target_arr, dum_arr), axis=1)
# print(dic)
numlis = np.arange(clumped_arr.shape[0])
rng.shuffle(numlis)
clumped_arr = clumped_arr[numlis]
# clumped_arr = clumped_arr[ numlis ]
clumped_target = clumped_arr[:]
ann = int((3/4)*clumped_target.shape[0])
print(ann)
tup_t = (target_rest_arr, target_rest_label), (target_test_arr, target_test_label) = (clumped_target[:ann, :-1], clumped_target[:ann, -1:]), (clumped_target[ann:, :-1], clumped_target[ann:, -1:])
#print(tup_t)
fs = open(pstri + "tar_tup.pickle", "wb")
pickle.dump(tup_t, fs)
fs.close()
target_dim = target_rest_arr.shape[1]
fs = open(pstri + "src_data_with_dic.pickle", "rb")
tup_s = pickle.load(fs)
fs.close()
source_arr, source_label, source_dic = tup_s
dum_arr = source_label.reshape((source_label.shape[0], 1))
clumped_arr = np.concatenate((source_arr, dum_arr), axis=1)
# print(dic)
numlis = np.arange(clumped_arr.shape[0])
rng.shuffle(numlis)
clumped_arr = clumped_arr[numlis]
# clumped_arr = clumped_arr[ numlis ]
clumped_source = clumped_arr[:]
#ann = (3//4)*clumped_source.shape[0]
ann  = clumped_source.shape[0]
tup_s = (source_rest_arr, source_rest_label), (source_test_arr, source_test_label) = (clumped_source[:ann, :-1], clumped_source[:ann, -1:]), (clumped_source[ann:, :-1], clumped_source[ann:, -1:])
fs = open(pstri + "src_tup.pickle", "wb")
pickle.dump(tup_s, fs)
fs.close()
source_dim = source_rest_arr.shape[1]


def dist(transformed_target, source_instance):
	return np.sqrt(np.sum((transformed_target - source_instance)**2))

def closeness_cost(W):
	sumi = 0
	for target_instance, target_lab in zip(target_rest_arr, target_rest_label):
		min_dist = np.inf
		target_instance = np.reshape(target_instance, (target_instance.shape[0], 1))
		transformed_target = np.dot(W, target_instance)
		transformed_target = np.ravel(transformed_target)
		for source_instance, source_lab in zip(source_rest_arr, source_rest_label):
			if source_lab == target_lab:
				min_dist = min(min_dist, dist(transformed_target, source_instance))
		sumi += min_dist
	return (sumi, np.sum(abs(W - np.identity(W.shape[0], 'float32'))))


x = tf.placeholder('float',[target_rest_arr.shape])
y = tf.placeholder('float')

def main():
	W = np.identity(source_dim)
	cost = closeness_cost(W)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(epochs):
		   opti, cost = sess.run([optimizer, cost], feed_dict = {x:target_rest_arr, y:target_rest_label})



if __name__ == '__main__':
	main()