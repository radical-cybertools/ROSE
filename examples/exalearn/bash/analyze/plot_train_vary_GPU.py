import matplotlib.pyplot as plt

# Data points
x = [1, 2, 3, 4, 8, 12]
y_orig = [6.94105, 3.62922, 2.5476, 1.94429, 2.19065, 1.99064]
y = [y_orig[0] / s for s in y_orig]

# Creating the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', linestyle='-', color='b')  # Blue line with circle markers
plt.title('Speed up vs number of ranks (one GPU per rank) used in training', fontsize=20)
plt.xlim(0, 13)
plt.xlabel('Number of rank', fontsize=16)
plt.ylabel('Speed up', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)

plt.savefig("train_vary_GPU.png")
plt.close()
