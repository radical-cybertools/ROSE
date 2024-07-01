import matplotlib.pyplot as plt
import numpy as np
import glob
path  = "/home/pb8294/data/TwoSourcesRdm/train/"

dist_files = glob.glob(path + "dist_*.txt")
indices = []
dist = []
srccd_x = []
srccd_y = []
snkcd_x = []
snkcd_y = []
srcint = []
snkint = []
big_data = []
for f in dist_files:
    split_file = f.split("_")[-1].split(".")[0]
    indices.append(split_file)


    x = np.absolute(np.loadtxt(path + "Cell_{}.dat".format(split_file)).astype(np.float32).reshape(100, 100))
    file3 = np.loadtxt(path + "dist_" + str(split_file) + ".txt")
    file4 = np.loadtxt(path + "srccd_" + str(split_file) + ".txt")
    file5 = np.loadtxt(path + "snkcd_" + str(split_file) + ".txt")
    file6 = np.loadtxt(path + "srcint_" + str(split_file) + ".txt")
    file7 = np.loadtxt(path + "snkint_" + str(split_file) + ".txt")

    dist.append(file3)
    srccd_x.append(file4[0])
    srccd_y.append(file4[1])
    snkcd_x.append(file5[0])
    snkcd_y.append(file5[1])
    srcint.append(file6)
    snkint.append(file7)
    big_data.append([file3, file4[0], file4[1], file5[0], file5[1], file6, file7])
    # print([file3, file4[0], file4[1], file5[0], file5[1], file6, file7])
    print(x.shape, file4[0], file4[1], file5[0], file5[1])
    plt.imshow(x)
    # plt.imshow(x[int(file4[0]) - 5 : int(file4[0]) + 6, int(file4[1]) - 5 : int(file4[1]) + 6])
    # plt.scatter(file4[1], file4[0])
    # plt.scatter(file5[1], file5[0])
    plt.savefig("zzzzz.png")
    exit(0)
    if file4[0] - 5 == 0 or file4[1] - 5 == 0 or file4[0] + 5 == 100 or file4[1] + 5 == 100:
        print("file4 0 or 100", file4)
    
    if file5[0] - 5 == 0 or file5[1] - 5 == 0 or file5[0] + 5 == 100 or file5[1] + 5 == 100:
        print("file5 0 or 100", file5)
big_data = np.array(big_data)

exit(0)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler = scaler.fit(big_data)
big_data_scaled = scaler.transform(big_data)


# print(len(big_data_scaled)); exit(0)
for jj, ind in enumerate(indices):
    print(jj, ind)
    np.savetxt(path + "norm_dist_{}.txt".format(ind), np.array([big_data_scaled[jj][0]]))
    np.savetxt(path + "norm_snkcd_{}.txt".format(ind), np.array([big_data_scaled[jj][1], big_data_scaled[jj][2]]))
    np.savetxt(path + "norm_srccd_{}.txt".format(ind), np.array([big_data_scaled[jj][3], big_data_scaled[jj][4]]))
    np.savetxt(path + "norm_srcint_{}.txt".format(ind), np.array([big_data_scaled[jj][5]]))
    np.savetxt(path + "norm_snkint_{}.txt".format(ind), np.array([big_data_scaled[jj][6]]))

for jj, ind in enumerate(indices):
    dd = np.loadtxt(path + "norm_dist_{}.txt".format(ind))
    print(big_data_scaled[jj][0], dd)
    if jj == 10:
        break



