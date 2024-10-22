import matplotlib.pyplot as plt

# Data points
x = [1, 2, 3, 4, 5, 6, 7, 8]
y = [9.41339, 6.94105, 6.91603, 6.96235, 6.95572, 6.93711, 6.90268, 6.94778]

# Creating the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', linestyle='-', color='b')  # Blue line with circle markers
plt.title('Average training time per epoch vs number of CPU cores in training', fontsize=20)
plt.xlabel('Number of CPU cores', fontsize=16)
plt.ylabel('Training time per epoch (s)', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)

plt.savefig("train_vary_CPU_cores.png")
plt.close()
