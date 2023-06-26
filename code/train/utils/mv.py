import os 

cwd = os.getcwd()
npy_files = list(os.listdir(cwd))
filtered_npy_files = [f for f in npy_files if f.endswith('.npy')]

num_train = int(0.9 * len(filtered_npy_files))

train_files = filtered_npy_files[:num_train]
test_files = filtered_npy_files[num_train:]

for f in train_files:
    os.system(f"cp {f} train/")

for f in test_files:
    os.system(f"cp {f} test/")
