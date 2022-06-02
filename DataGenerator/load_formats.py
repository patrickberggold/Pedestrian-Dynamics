import time
import h5py
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

root_dir = 'C:\\Users\\ga78jem\\Documents\\Revit\\Exports\\0__floorplan_zPos_0.0_roomWidth_0.24_numRleft_2.0_numRright_2.0\\pickle_variations'
assert os.path.isdir(root_dir)

filename = 'HDF5_floorplan_zPos_0.0_roomWidth_0.24_numRleft_2.0_numRright_2.0_variation_0.h5'
filename = os.path.join(root_dir, filename)
assert os.path.isfile(filename)

# Load HDF5
overall_time = time.time()
for filename in os.listdir(root_dir):
    if filename.startswith('HDF5'):
        start_time = time.time()
        hf = h5py.File(os.path.join(root_dir, filename), 'r')
        n1 = hf.get('img')
        print('HDF5: ', time.time()-start_time)
print('HDF5 IN TOTAL:', time.time()-overall_time)

plt.axis('off')
# plt.imshow(img, vmin=0, vmax=255)

# Load Numpy
# overall_time = time.time()
# for filename in os.listdir(root_dir):
#     if filename.startswith('NUMPY'):
#         start_time = time.time()
#         arr = np.load(os.path.join(root_dir, filename))
#         print('NUMPY: ', time.time()-start_time)
# print('NUMPY IN TOTAL:', time.time()-overall_time)

# Load Pickle
# overall_time = time.time()
# for filename in os.listdir(root_dir):
#     if filename.startswith('PICKLE'):
#         start_time = time.time()
#         with open(os.path.join(root_dir, filename),'rb') as f:
#             x = pickle.load(f)
#         print('PICKLE: ', time.time()-start_time)
# print('PICKLE IN TOTAL:', time.time()-overall_time)
