import matplotlib.pyplot as plt

#It should be 1, 2 or 3
#AL_phase_idx=2
# Data setup
centers = [0.000190619708,          9.01861632e-05,       4.96659404e-05]  # Central values of each band
errors =  [5.559355422743267e-05,   3.22970222452586e-05, 1.3117838937239361e-05]  # Errors or half-widths of each band


# Additional data for vertical bars
points_orig =        [4500,                  18000,               36000, 54000, 72000]  # X positions for the vertical bars
points = [x * 3 for x in points_orig]

point_centers = [0.00107867866,         0.000330467698,        0.0002030424228, 0.00020329361166666665, 0.000132865114]  # Y positions for each point
point_errors =  [0.0003555528552316384, 3.788501125753794e-05, 5.201472722198465e-05, 3.864500095366032e-05, 4.2541479765770065e-05]  # Errors for each point

# Plot setup
plt.figure(figsize=(12, 8))

center = centers[0]
error = errors[0]
plt.hlines(y=center + error, xmin=0, xmax=240000, color='red', linewidth=2, label="Error band from training with AL, phase 1")
plt.hlines(y=center - error, xmin=0, xmax=240000, color='red', linewidth=2)

center = centers[2]
error = errors[2]
plt.hlines(y=center + error, xmin=0, xmax=240000, color='blue', linewidth=2, label="Error band from training with AL, phase 3")
plt.hlines(y=center - error, xmin=0, xmax=240000, color='blue', linewidth=2)


for x, y, e in zip(points, point_centers, point_errors):
    plt.errorbar(x, y, yerr=e, fmt='o', color='black', capsize=5, label="Error bar from bulk training")

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # Creating a dictionary to remove duplicates
plt.legend(by_label.values(), by_label.keys(), fontsize=16)

# Customization
#plt.ylim(0, 0.0025)
plt.xlim(0, 240000)
plt.xlabel('Number of samples in bulk training', fontsize=16)
plt.ylabel('l2-difference', fontsize=16)
plt.title('l2-difference from AL training phase 1 and 3 and bulk training', fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.show()
plt.savefig("l2-difference-AL-vs-bulk")
plt.close()
