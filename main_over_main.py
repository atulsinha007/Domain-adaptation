import os
import pickle
import sys
with open("./log_folder/dublue_pareto_set_of_pareto.pickle", "rb") as fp:
	lis = pickle.load(fp)

for i in range(len(lis)):
	os.system("python3 main.py _pareto_set_"+str(i) + " "+str(i))