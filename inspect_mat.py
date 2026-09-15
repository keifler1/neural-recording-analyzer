from scipy.io import loadmat

mat = loadmat("data/C_Easy2_noise015.mat")

print("Keys found in the file:")
for key in mat.keys():
    if not key.startswith("__"):
        print(f"  {key}: shape {mat[key].shape}, dtype {mat[key].dtype}")