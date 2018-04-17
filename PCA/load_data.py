import cv2
import os
import numpy as np

TRAIN_DIR = '/home/ak/Documents/forgit/Domain-adaptation/PCA/data_set/amazon/'
TEST_DIR = '/home/ak/Documents/forgit/Domain-adaptation/PCA/data_set/dslr/'

train_dir = '/home/ak/Documents/forgit/Domain-adaptation/PCA/Generated/amazon/'
test_dir = '/home/ak/Documents/forgit/Domain-adaptation/PCA/Generated/dslr/'


def create_dataset(fileList, domain, dir_name):

	for i, file in enumerate(fileList):
		oldfile = file
		file = os.path.join(dir_name, file)
		img = cv2.imread(file)
		resized_img = cv2.resize(img, (100,100))
		resized_img = cv2.cvtColor( resized_img, cv2.COLOR_RGB2GRAY )
		cv2.imwrite('/home/ak/Documents/forgit/Domain-adaptation/PCA/Generated/' + domain + '/'+ oldfile + '_resized_bw',resized_img)
		

def create():	

	fileList = sorted(os.listdir(TRAIN_DIR))
	create_dataset(fileList, "amazon", TRAIN_DIR)
	fileList = sorted(os.listdir(TEST_DIR))
	create_dataset(fileList, "dslr", TEST_DIR)


def read_data(fileList, dir_name):
	
	x = []
	y = []
	for i, file in enumerate(fileList):
		oldfile = file
		file = os.path.join(dir_name, file)
		img = cv2.imread(file)
		img =  cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )
		data_reshaped = img.flatten()
		y.append(int(oldfile[0]))
		x.append(data_reshaped)
		# print(data_reshaped.shape)
	y = np.asarray(y)
	y = y.reshape(-1, 1)
	x = np.asarray(x)
	x = x.transpose()
	return x, y



def load_data():

	fileList = sorted(os.listdir(train_dir))
	trainx, trainy = read_data(fileList, train_dir)
	fileList = sorted(os.listdir(test_dir))
	testx, testy = read_data(fileList, test_dir)
	return trainx, trainy, testx, testy


def main():
	create()

if __name__ == '__main__':
	main()