import matplotlib.pyplot as plt
import numpy as np
non = [219000910.41, 423532079.662, 635172620.979, 584221749.29, 481051241.357]
imposter = [743484568.726, 679907245.069, 697097729.246, 1132549795.7, 699185954.616]

non.sort()
imposter.sort()
t1 = np.arange(len(non))
t2 = np.arange(len(imposter))
plt.plot(t1, non, color = 'black', label = 'non imposter')
plt.plot(t2, imposter, color = 'red', label = 'imposter')
plt.xlabel('Images')
plt.ylabel('Distance')
plt.title('PCA')
plt.legend()
plt.show()
print()