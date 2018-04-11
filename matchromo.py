class MatChromo:
	def __init__(self, source_dim, target_dim, arr=None, rng = np.random):
		self.array = rng.random((source_dim, target_dim))

	def closeness_cost(self):
		ind = self
		sumi = 0
		W = ind.array
		# W = np.identity(W.shape[0])
		# print( source_dic)

		# print(target_rest_label)
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