import array
import random
import time
import numpy
from math import sqrt
import cluster
from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence
from deap import creator
from deap import tools
import os
from population import *
from network import Neterr
from chromosome import Chromosome, crossover
import traceback
n_hidden = 100
indim = 32
outdim = 5



network_obj_tar = Neterr(indim, outdim, n_hidden, change_to_target=1, rng=random)
# creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, 0.0, 0.0))
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual", Chromosome, fitness=creator.FitnessMin)
print("here network object created")
toolbox = base.Toolbox()




def minimize_tar(individual):
	outputarr = network_obj_tar.feedforward_ne(individual,  final_activation=network.softmax)

	neg_log_likelihood_val = give_neg_log_likelihood(outputarr, network_obj_tar.resty)
	mean_square_error_val = give_mse(outputarr, network_obj_tar.resty)
	complexity = lambda ind: len(ind.conn_arr) * ind.node_ctr
	ind_complexity = complexity(individual)
	# anyways not using these as you can see in 'creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, 0.0, 0.0))'
	# return neg_log_likelihood_val, mean_square_error_val, false_positve_rat, false_negative_rat
	return neg_log_likelihood_val, mean_square_error_val, ind_complexity


def mycross(ind1, ind2, gen_no):
	child1 = crossover(ind1, ind2, gen_no, inputdim=indim, outputdim=outdim)
	child2 = crossover(ind1, ind2, gen_no, inputdim=indim, outputdim=outdim)

	return child1, child2


def mymutate(ind1):
	new_ind = ind1.do_mutation(rate_conn_weight=0.2,rate_conn_itself= 0.1,rate_node= 0.05,weight_factor = 1, inputdim= indim, outputdim=outdim, max_hidden_unit=n_hidden,rng = random)
	return new_ind


def initIndividual(ind_class, inputdim, outputdim):
	ind = ind_class(inputdim, outputdim)
	return ind

old_chromosome = None
toolbox.register("individual", initIndividual, creator.Individual, indim, outdim)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", mycross)
toolbox.register("mutate", mymutate)
toolbox.register("select", tools.selNSGA2)

bp_rate = 0.05


def main(seed=None, play=0, NGEN=40, MU=4 * 10):
	# random.seed(seed)

	# MU has to be a multiple of 4. period.
	CXPB = 0.9

	stats = tools.Statistics(lambda ind: ind.fitness.values[1])
	# stats.register("avg", numpy.mean, axis=0)
	# stats.register("std", numpy.std, axis=0)
	stats.register("min", numpy.min, axis=0)
	stats.register("max", numpy.max, axis=0)

	logbook = tools.Logbook()
	logbook.header = "gen", "evals", "std", "min", "avg", "max"
	toolbox.register("evaluate", minimize_tar)
	time1 = time.time()
	pop_tar = toolbox.population(n=MU)
	time2 = time.time()
	print("After population initialisation", time2 - time1)

	invalid_ind = [ind for ind in pop_tar if not ind.fitness.valid]

	fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
	for ind, fit in zip(invalid_ind, fitnesses):
		ind.fitness.values = fit

	# This is just to assign the crowding distance to the individuals
	# no actual selection is done
	pop_tar = toolbox.select(pop_tar, len(pop_tar))
	# print( "first population selected, still outside main loop")
	# print(pop)
	record = stats.compile(pop_tar)
	logbook.record(gen=0, evals=len(invalid_ind), **record)
	print(logbook.stream)
	maxi = 0
	stri = ''
	flag = 0
	# Begin the generational process
	# print(pop.__dir__())
	for gen in range(1, NGEN):

		# Vary the population
		print()
		print("here in gen no.", gen)
		offspring = tools.selTournamentDCD(pop_tar, len(pop_tar))
		offspring = [toolbox.clone(ind) for ind in offspring]
		if play:
			if play == 1:
				pgen = NGEN * 0.1
			elif play == 2:
				pgen = NGEN * 0.9

			if gen == int(pgen):
				print("gen:", gen, "doing clustering")
				to_bp_lis = cluster.give_cluster_head(offspring, int(MU * bp_rate))
				assert (to_bp_lis[0] in offspring)
				print("doing bp")
				[item.modify_thru_backprop(indim, outdim, network_obj_tar.rest_setx, network_obj_tar.rest_sety,
										   epochs=10, learning_rate=0.1, n_par=10) for item in to_bp_lis]
				# Evaluate the individuals with an invalid fitness
				invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
				fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
				for ind, fit in zip(invalid_ind, fitnesses):
					ind.fitness.values = fit
		dum_ctr = 0
		for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
			# print(ind1.fitness.values)
			"""if not flag :
				ind1.modify_thru_backprop(indim, outdim, network_obj.rest_setx, network_obj.rest_sety, epochs=10, learning_rate=0.1, n_par=10)
				flag = 1
				print("just testing")
			"""
			flag = 0
			if random.random() <= CXPB:
				ind1, ind2 = toolbox.mate(ind1, ind2, gen)
				ind1 = creator.Individual(indim, outdim, ind1)
				ind2 = creator.Individual(indim, outdim, ind2)
				flag = 1
			maxi = max(maxi, ind1.node_ctr, ind2.node_ctr)
			toolbox.mutate(ind1)
			toolbox.mutate(ind2)

			offspring[dum_ctr] = ind1
			offspring[dum_ctr + 1] = ind2
			del offspring[dum_ctr].fitness.values, offspring[dum_ctr + 1].fitness.values
			dum_ctr += 2

		# Evaluate the individuals with an invalid fitness
		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit

		# Select the next generation population
		pop_tar = toolbox.select(pop_tar + offspring, MU)

		record = stats.compile(pop_tar)
		logbook.record(gen=gen, evals=len(invalid_ind), **record)
		anost = logbook.stream
		liso = [item.rstrip() for item in anost.split("\t")]
		mse = float(liso[3])

		print(anost)
		stri += anost + '\n'
		print("generation done")
		# file_ob.write(str(logbook.stream))
		# print(len(pop))
		# file_ob.close()
	# print(stri)

	##from here starting target
	return pop_tar, logbook


def note_this_string(new_st, stringh):
	"""flag_ob = open("flag.txt","r+")

    ctr = None
    st = flag_ob.read()
    flag = int(st.rstrip())
    while flag ==1:
        flag_ob.seek(0)
        st = flag_ob.read()
        flag = int(st.rstrip())
        time.sleep(3)
    if flag == 0:
        flag = 1
        flag_ob.seek(0)
        flag_ob.write("1\n")
        flag_ob.close()
        '/home/robita/forgit/neuro-evolution/05/state/tf/indep_pima/input/model.ckpt.meta'
    """
	name = "./ctr_folder/ctr" + stringh + ".txt"
	if not os.path.isfile(name):
		new_f = open(name, "w+")
		new_f.write("0\n")
		new_f.close()

	ctr_ob = open(name, "r+")
	strin = ctr_ob.read().rstrip()
	assert (strin is not '')
	ctr = int(strin)
	ctr_ob.seek(0)
	ctr_ob.write(str(ctr + 1) + "\n")
	ctr_ob.close()
	"""  
        flag_ob = open("flag.txt","w")
        flag_ob.write("0\n")
        flag_ob.close()
    """

	new_file_ob = open("log_folder/log" + stringh + ".txt", "a+")
	new_file_ob.write(str(ctr) + " " + new_st + "\n")
	new_file_ob.close()
	return ctr




def test_it_with_bp(play=1, NGEN=100, MU=4 * 25, play_with_whole_pareto=0):
	pop, stats = main(play=play, NGEN=NGEN, MU=MU)
	stringh = "_with_bp_just_tar" + str(play) + "_" + str(NGEN)
	fronts = tools.sortNondominated(pop, len(pop))

	'''file_ob = open("./log_folder/log_for_graph.txt", "w+")
    for item in fronts[0]:
        st = str(item.fitness.values[0]) + " " + str(item.fitness.values[1])+"\n"
        file_ob.write( st )
    file_ob.close()'''

	if play_with_whole_pareto or len(fronts[0]) < 30:
		pareto_front = fronts[0]
	else:

		pareto_front = random.sample(fronts[0], 30)
	print("Pareto Front: ")
	for i in range(len(pareto_front)):
		print(pareto_front[i].fitness.values)

	print("\ntest: test on one with min validation error",
		  network_obj_tar.test_err(min(pop, key=lambda x: x.fitness.values[1])))
	tup = network_obj_tar.test_on_pareto_patch_correctone(pareto_front)

	print("\n test: avg on sampled pareto set", tup)

	st = str(network_obj_tar.test_err(min(pop, key=lambda x: x.fitness.values[1]))) + " " + str(tup)
	print(note_this_string(st, stringh))


if __name__ == "__main__":
	logf = open("log_error_just_tar.txt", "a")
	try:
		test_it_with_bp(play=1, NGEN=100, MU=4 * 25, play_with_whole_pareto=1)
	except Exception as e:
		print("Error! Error! Error!")
		logf.write('\n\n')
		localtime = time.localtime(time.time())
		logf.write(str(localtime)+'\n')
		traceback.print_exc(file=logf)
		logf.write('\n\n')
	finally:
		logf.close()
	# file_ob.write( "test on one with min validation error " + str(neter.test_err(min(pop, key=lambda x: x.fitness.values[1]))))

	# print(stats)
	'''
    import matplotlib.pyplot as plt
    import numpy

    front = numpy.array([ind.fitness.values for ind in pop])
    plt.scatter(front[:,0], front[:,1], c="b")
    plt.axis("tight")
    plt.show()'''
