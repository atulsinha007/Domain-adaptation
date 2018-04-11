import numpy as np
class MatChromo:
	def __init__(self, source_dim, target_dim, arr=None, rng = np.random):
		self.array = rng.random((source_dim, target_dim))

	def closeness_cost(self, network_obj):
		ind = self
		sumi = 0
		W = ind.array
		# W = np.identity(W.shape[0])
		# print( source_dic)

		# print(target_rest_label)
		target_rest_arr, target_rest_label = network_obj.seventyfive_target_set
		(source_rest_arr, source_rest_label) = network_obj.full_source_set
		for target_instance, target_lab in zip(target_rest_arr, target_rest_label):
			# assert (target_instance in source_rest_arr)

			min_dist = np.inf
			target_instance = np.reshape(target_instance, (target_instance.shape[0], 1))

			transformed_target = np.dot(W, target_instance)
			transformed_target = np.ravel(transformed_target)

			# assert (transformed_target in source_rest_arr)
			# print(source_rest_arr.shape)
			for source_instance, source_lab in zip(source_rest_arr, source_rest_label):
				# print("here")
				# print(transformed_target, source_instance)
				# print(source_instance)
				if source_lab == target_lab:
					min_dist = min(min_dist, dist(transformed_target, source_instance))
			# print(min_dist)
			sumi += min_dist

		# sumi += np.sum(abs(W - np.identity(W.shape[0], 'float32')))
		# print( sumi )
		return (sumi, np.sum(abs(W - np.identity(W.shape[0], 'float32'))))

	def Mutate(self, m_prob, m_fac=1, rng=np.random):
		arr = self.array
		for row in range(arr.shape[0]):
			index = rng.randint(0, arr.shape[1])
			if rng.random() < m_prob:
				arr[row][index] += rng.uniform(-1, 1) * m_fac
		del (self.fitness.values)

def Crossover(ind1, ind2, cxpb  = 1, rng = np.random):
	arr1 = ind1.array
	arr2 = ind2.array
	for row in range(arr1.shape[0]):
		for col in range(arr1.shape[1]):
			if rng.random() < cxpb:
				alpha = rng.random()
				temp = copy.deepcopy(arr1[row][col])
				arr1[row][col] = alpha*arr1[row][col] + (1-alpha)*arr2[row][col]
				arr2[row][col] = alpha*arr2[row][col] + (1-alpha)*temp
	del ind1.fitness.values
	del ind2.fitness.values
	return ind1, ind2