import matplotlib.pyplot as plt

# Data points
x = [8, 16, 24, 32, 48, 64, 96]
y_orig = [357757, 213400, 147625, 118495, 96821, 84848, 72177]
y = [y_orig[0] / s for s in y_orig]

# Creating the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', linestyle='-', color='b')  # Blue line with circle markers
plt.title('Speed up vs number of ranks (one CPU core per rank) used in training', fontsize=20)
plt.xlim(0, 100)
plt.xlabel('Number of ranks',  fontsize=16)
plt.ylabel('Speed up',  fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)

plt.savefig("sim_vary_CPU_cores.png")
plt.close()
