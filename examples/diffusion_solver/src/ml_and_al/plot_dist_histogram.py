import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 36})
path  = "/home/pb8294/data/TwoSourcesRdm/train/"
dist_files = glob.glob(path + "dist_*.txt")
dists = []
for i in range(len(dist_files)):
    dists.append(np.loadtxt(dist_files[i]).tolist())
dists = np.array(dists)

plt.figure(figsize=(20, 20))
plt.hist(dists, bins=100)
plt.xlabel("Distance between sources")
plt.ylabel("Frequency")
plt.title("Histogram of distance between sources")
plt.savefig("./distrib/plot_dist_histogram_train.png", transparent=True)


dist_files = glob.glob(path + "snkcd_*.txt")
dists = []
for i in range(len(dist_files)):
    dists.append(np.loadtxt(dist_files[i]).tolist())
dists = np.array(dists)
plt.figure(figsize=(24, 20), dpi=300)
plt.hist2d(dists[:, 1], dists[:, 0], bins=100)
plt.colorbar()
plt.xlabel("Source 1 x-coord")
plt.ylabel("Source 1 y-coord")
plt.title("Histogram of Source 1 position")
plt.savefig("./distrib/plot_snkcoord_histogram_train.png", transparent=True)

dist_files = glob.glob(path + "srccd_*.txt")
dists = []
for i in range(len(dist_files)):
    dists.append(np.loadtxt(dist_files[i]).tolist())
dists = np.array(dists)
plt.figure(figsize=(24, 20), dpi=300)
plt.hist2d(dists[:, 1], dists[:, 0], bins=100)
plt.colorbar()
plt.xlabel("Source 2 x-coord")
plt.ylabel("Source 2 y-coord")
plt.title("Histogram of Source 2 position")
plt.savefig("./distrib/plot_srccoord_histogram_train.png", transparent=True)


# exit()

dist_files = glob.glob(path + "Cell_*.dat")
dists = []
# print(dist_files)
src_intensity = []
snk_intensity = []
vals = np.arange(len(dist_files))
# print(vals)
np.random.shuffle(vals)

for i in range(len(dist_files)):
# for i in vals[:10]:
    split_file = (dist_files[i].split("_")[-1]).split(".")
    x = np.absolute(np.loadtxt(dist_files[i]).astype(np.float32).reshape(100, 100))
    src_coord = np.loadtxt(path + "srccd_{}.txt".format(split_file[0])).astype(np.int64).tolist()
    snk_coord = np.loadtxt(path + "snkcd_{}.txt".format(split_file[0])).astype(np.int64).tolist()
    # print(i, src_coord, snk_coord)

    src_int = x[src_coord[0], src_coord[1]]
    snk_int = x[snk_coord[0], snk_coord[1]]

    # start = src_coord[0]
    # end = src_coord[0]
    # while src_int != 0:
    #     start += 1
    #     src_int = x[start, src_coord[1]]

    # print(split_file[0], end - start)
    # exit(0)
    src_intensity.append(src_int)
    snk_intensity.append(snk_int)

    # np.savetxt(path + "srcint_{}.txt".format(split_file[0]), np.array([src_int]))
    # np.savetxt(path + "snkint_{}.txt".format(split_file[0]), np.array([snk_int]))

    # print(split_file[0], src_int, snk_int,)
    
    # plt.close("all")
    # plt.figure(figsize=(12, 10), dpi=300)
    # print(split_file[0], src_int, snk_int)
    # plt.imshow(x)
    # plt.xticks(np.arange(0, 100, 5))
    # plt.scatter(src_coord[1], src_coord[0], color="red")
    # plt.scatter(snk_coord[1], snk_coord[0], color="red")
    
    # plt.savefig("z_{}.png".format(split_file[0]))

plt.figure(figsize=(20, 20))
plt.hist(src_intensity, bins=100)
plt.xlabel("Source Intensity")
plt.ylabel("Frequency")
plt.title("Histogram of Source intensity")
plt.savefig("./distrib/plot_src_intensity_histogram_train.png")

plt.figure(figsize=(20, 20), dpi=300)
plt.hist(snk_intensity, bins=100)
plt.xlabel("Intensity of Source 2")
plt.ylabel("Frequency")
plt.title("Histogram of Sink intensity")
plt.savefig("./distrib/plot_snk_intensity_histogram_train.png", transparent=True)
