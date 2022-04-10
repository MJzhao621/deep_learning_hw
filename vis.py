import matplotlib.pyplot as plt
import numpy as np
import pickle

save_path = "./save/weights.pkl"
weights = open(save_path, "rb")
weights = pickle.load(weights)

plt.imshow(weights["output_weight"], interpolation='nearest', cmap='bone', origin='upper')
plt.show()