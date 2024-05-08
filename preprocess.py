import h5py
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


# Process the first mormyrus pair
with h5py.File('/Users/olha/Study/Continual Learning/fish_pairs/Mormyrus_Pair_01/poses/20230316_Mormyrus_Pair_01.000_20230316_Mormyrus_Pair_01.analysis.h5', 'r') as f:
    dset_names = list(f.keys())
    locations = f["tracks"][:].T
    node_names = [n.decode() for n in f["node_names"][:]]

print("===HDF5 datasets===")
print(dset_names)
print()

print("===locations data shape===")
print(locations.shape)
print()

print("===nodes===")
for i, name in enumerate(node_names):
    print(f"{i}: {name}")
print()


# Visualize the first fish
HEAD_INDEX = 0
MIDDLE_INDEX = 5
BACK_INDEX = 4

head_loc = locations[:, HEAD_INDEX, :, :]
middle_loc = locations[:, MIDDLE_INDEX, :, :]
back_loc = locations[:, BACK_INDEX, :, :]

sns.set('notebook', 'ticks', font_scale=1.2)
mpl.rcParams['figure.figsize'] = [15,6]

plt.figure()
plt.plot(head_loc[:,0,0], 'y',label='fish-1')
plt.plot(head_loc[:,0,1], 'g',label='fish-2')

plt.plot(-1*head_loc[:,1,0], 'y')
plt.plot(-1*head_loc[:,1,1], 'g')

plt.legend(loc="center right")
plt.title('Head locations')


plt.figure(figsize=(7,7))
plt.plot(head_loc[:,0,0],head_loc[:,1,0], 'y',label='fish-1')
plt.plot(head_loc[:,0,1],head_loc[:,1,1], 'g',label='fish-2')
plt.legend()

plt.xlim(0,2048)
plt.xticks([])

plt.ylim(0,2048)
plt.yticks([])
plt.title('Head tracks')

plt.show()