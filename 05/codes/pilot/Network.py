
import numpy as np
#import tf_mlp
#import tensorflow as tf
import time
import gene
import chromosome

from chromosome import *

def sigmoid(arr):
    return 1 / (1 + np.exp(-arr))
def relu(arr):
    return np.where(arr>0,arr,0)
def priortize_connections(conn_lis):
    dict={'IH1':[],
          'H1H2':[],
          'IH2':[],
          'H2O':[],
          'H1O':[],
          'IO':[]
          }
    for concsn in conn_lis:
        tup=concsn.get_couple()
        dict[tup[0].nature+tup[1].nature].append(concsn)
    return dict['IH1']+['breakH1']+dict['H1H2']+dict['IH2']+['breakH2']+dict['H2O']+dict['H1O']+dict['IO']
class Neterr:
    def __init__(self, inputdim, outputdim,inputarr,  hidden_unit_lim ,rng):
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.inputarr=inputarr  #self explanatory
        self.hidden_unit_lim = hidden_unit_lim
        self.rng = rng

        #self.arr_of_net = arr_of_net
    """
    def set_arr_of_net(self, newarr_of_net):
        self.arr_of_net = newarr_of_net
    """

    def feedforwardcm(self, inputdim, outputdim, inputarr, rng):
        ConnMatrix = {}             #Connection Matrix
        WeightMatrix = {}           #Weight Matrix
        NatureCtrDict = {}          #Contains Counter of Nature { 'I', 'H1', 'H2', 'O' }
        NatureCtrDict['I'] = 0
        NatureCtrDict['H1'] = 0
        NatureCtrDict['H2'] = 0
        NatureCtrDict['O'] = 0
        
        dictionary = {}             #Contains node numbers mapping starting from 0, nature-wise
        dictionary['I'] = {}
        dictionary['H1'] = {}
        dictionary['H2'] = {}
        dictionary['O'] = {}
        age = -1
        chromo = Chromosome(age)

        for i in chromo.node_arr:
            dictionary[i.nature][i.node_num] = NatureCtrDict[i.nature]
            NatureCtrDict[i.nature] += 1
        
        ConnMatrix['IO'] = np.zeros((inputdim, outputdim))
        ConnMatrix['IH1'] = np.zeros((inputdim, NatureCtrDict['H1']))
        ConnMatrix['IH2'] = np.zeros((inputdim, NatureCtrDict['H2']))
        ConnMatrix['H1H2'] = np.zeros((NatureCtrDict['H1'], NatureCtrDict['H2']))
        ConnMatrix['H1O'] = np.zeros((NatureCtrDict['H1'], outputdim))
        ConnMatrix['H2O'] = np.zeros((NatureCtrDict['H2'], outputdim))

        WeightMatrix['IO'] = np.zeros((inputdim, outputdim))
        WeightMatrix['IH1'] = np.zeros((inputdim, NatureCtrDict['H1']))
        WeightMatrix['IH2'] = np.zeros((inputdim, NatureCtrDict['H2']))
        WeightMatrix['H1H2'] = np.zeros((NatureCtrDict['H1'], NatureCtrDict['H2']))
        WeightMatrix['H1O'] = np.zeros((NatureCtrDict['H1'], outputdim))
        WeightMatrix['H2O'] = np.zeros((NatureCtrDict['H2'], outputdim))

        for con in chromo.conn_arr:
            ConnMatrix[con.source.nature + con.destination.nature][dictionary[con.source.nature][con.source.node_num]][dictionary[con.destination.nature][con.destination.node_num]] = 1
            WeightMatrix[con.source.nature + con.destination.nature][dictionary[con.source.nature][con.source.node_num]][dictionary[con.destination.nature][con.destination.node_num]] = con.weight 

        inputarr = self.rng.random((inputarr.shape[0], inputarr.shape[1]))
        print(inputarr)
        print("---------------")
        ConnMatrix['IO'] = self.rng.random((inputdim, outputdim))
        print(ConnMatrix['IO'])
        print("---------------")
        OUTPUT = np.dot(inputarr, ConnMatrix['IO'])
        print(OUTPUT)
        


    def feedforward_ne(self,chromosome,middle_activation=relu,final_activation=sigmoid):

        conn_list = priortize_connections(chromosome.conn_arr)  #list of connections with string type breaks to seperate
        return_arr=np.array([])
        for i in range(self.inputarr.shape[0]):

            storage = [0 for i in range(self.hidden_unit_lim + self.outputdim)]
            storage=np.array([0]+list(self.inputarr[i])+storage) #here [0] is dummy storage as we use '1' indexing for node_ctr

            node_num_lis=[]
            for connection in conn_list:

                if type(connection)==str:

                    for node_num in node_num_lis:
                        storage[node_num]=middle_activation(storage[node_num])
                    node_num_lis=[]
                    continue

                tup = connection.get_couple()
                node_num_lis.append(tup[1].node_num)
                weight = connection.__getattribute__('weight')
                storage[tup[1].node_num] += storage[tup[0].node_num]*weight
            #print(storage)
            bias_weights=[bn.weight for bn in chromosome.bias_conn_arr]
            for p in range(len(bias_weights)):
                storage[self.inputdim + 1+p]    += -1*bias_weights[p]
            output_part = storage[self.inputdim+1:self.outputdim+self.inputdim+1]
            return_arr = np.concatenate((return_arr,output_part))
        return final_activation(return_arr.reshape((self.inputarr.shape[0],self.outputdim)))


        #pass

    def test(self, weight_arr):
        pass

    def modify_thru_backprop(self, popul, epochs=10, learning_rate=0.01, L1_reg=0.00001, L2_reg=0.0001):
        pass


def squa_test(x):
    return (x ** 2).sum(axis=1)

def dummy_popultation(number):#return list of chromosomes
    chromolis=[]
    for i in range(number):
        newchromo=chromosome.Chromosome(0)
        newchromo.rand_init()
        chromolis.append(newchromo)
    return chromolis


def main1():
    indim = 4
    outdim = 1
    arr_of_net = np.zeros((4,4))
    np.random.seed(4)
    neter = Neterr(indim, outdim, arr_of_net, 10, np.random)
    neter.feedforwardcm(indim, outdim, arr_of_net, np.random)



def main():
    indim=4
    outdim=3
    np.random.seed(4)
    num_data=2
    inputarr=np.random.random((num_data,indim))
    neter = Neterr(indim, outdim, inputarr, 10, np.random)
    chromo=chromosome.Chromosome(0)
    chromo.rand_init(indim,outdim,np.random)
    print(neter.feedforward_ne(chromo))


if __name__ == '__main__':
    main()
