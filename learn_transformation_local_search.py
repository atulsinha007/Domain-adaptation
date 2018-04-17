#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.


#    example which maximizes the sum of a list of integers
#    each of which can be 0 or 1

import random
import numpy as np

from deap import base
from deap import creator
from deap import tools
import pickle
import copy
import sys
import numpy 
import time
NGEN = 300
pop_size = 4 * 50
cxpb = .8
m_fac = .2
m_prob = .2


np.random.seed(10)
rng = np.random
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

source_dim = 32
target_dim = 32
class MatChromo:
	def __init__(self, source_dim, target_dim, arr=None, rng = np.random):
		self.array = rng.random((source_dim, target_dim))



def return_obj(class_name, source_dim, target_dim):
	return class_name(source_dim, target_dim)


def dist(transformed_target, source_instance):
	return np.sqrt(np.sum((transformed_target - source_instance)**2))


def closeness_cost(ind):
	sumi = 0
	W = ind.array
	#W = np.identity(W.shape[0])
	# print( source_dic)

	#print(target_rest_label)
	for target_instance, target_lab in zip(target_rest_arr, target_rest_label):
		# assert (target_instance in source_rest_arr)
		
		min_dist = np.inf
		target_instance = np.reshape(target_instance, (target_instance.shape[0], 1))
		

		transformed_target = np.dot(W, target_instance)
		transformed_target = np.ravel(transformed_target)

		#assert (transformed_target in source_rest_arr)
		#print(source_rest_arr.shape)
		for source_instance, source_lab in zip(source_rest_arr, source_rest_label):
			#print("here")
			# print(transformed_target, source_instance)
			#print(source_instance)
			if source_lab == target_lab:
				min_dist = min(min_dist, dist(transformed_target, source_instance))
			#print(min_dist)
		sumi += min_dist

	# sumi += np.sum(abs(W - np.identity(W.shape[0], 'float32')))
	# print( sumi )
	return sumi


def myMutate(darr, m_prob = 1, m_fac = .01, rng = np.random):
	arr = darr.array
	for row in range(arr.shape[0]):
		index = rng.randint(0, arr.shape[1])
		if rng.random() < m_prob:
			arr[row][index] += rng.uniform(-1, 1)*m_fac
	#del(darr.fitness.values)

def ensemble(pop_lis):
	new_3d_arr = np.array([item.array for item in pop_lis])
	me = np.mean(new_3d_arr, axis = 0)
	assert ( me.shape == pop_lis[0].array.shape)
	return me
def pivot_search(N_iter = 1000):
	mat = MatChromo(source_dim, target_dim)
	mat.array = np.identity(source_dim)
	min_cost = closeness_cost(mat)
	best_ind = copy.deepcopy(mat)
	for i in range(N_iter):
		temp_ind = copy.deepcopy(mat)
		myMutate(mat)

		new_cost = closeness_cost(mat)
		if new_cost < min_cost:
			best_ind = copy.deepcopy(mat)
			min_cost = new_cost
			print("piv_min now", min_cost)
		mat = temp_ind
	return best_ind
def caterpillar_search(N_iter = 1000):

	new_mat = MatChromo(source_dim, target_dim)
	new_mat.array = np.identity(source_dim)
	min_cost = closeness_cost(new_mat)
	best_ind = copy.deepcopy(new_mat)
	for i in range(N_iter):

		myMutate(new_mat)
		new_cost = closeness_cost(new_mat)
		if new_cost < min_cost:

			best_ind = copy.deepcopy(new_mat)
			min_cost = new_cost
			print("Cater_min now", min_cost)
			print("Cater_min now", min_cost)
	return best_ind
def main():
	seedh = 1064
	N_iter = 1000
	random.seed(seedh)
	best_ind = caterpillar_search(N_iter = N_iter)
	with open("./pickle_jar/cat_search_dublue.pickle", "wb") as fp:
		pickle.dump(best_ind, fp)
	with open("./log_folder/cat_search_log.pickle", "a") as fp:
		fp.write(str(seedh)+" "+str(N_iter)+" "+str(closeness_cost(best_ind)))
	best_ind = pivot_search(N_iter= N_iter)
	with open("./pickle_jar/piv_search_dublue.pickle", "wb") as fp:
		pickle.dump(best_ind, fp)
	with open("./log_folder/piv_search_log.pickle", "a") as fp:
		fp.write(str(seedh) + " " + str(N_iter) + " " + str(closeness_cost(best_ind)))

if __name__ == "__main__":
	main()
