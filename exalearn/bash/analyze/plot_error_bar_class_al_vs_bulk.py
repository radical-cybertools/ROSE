import matplotlib.pyplot as plt

#It should be 1, 2 or 3
#AL_phase_idx=1
# Data setup
centers = [0.000407634226,          0.00018441775199999998,     7.3676936e-05]  # Central values of each band
errors =  [0.00015972274060767233,  6.452705198370165e-05,      5.934087042221745e-06]  # Errors or half-widths of each band


# Additional data for vertical bars
points_orig =        [4500,                  18000,               36000, 54000, 72000]  # X positions for the vertical bars
points = [x * 3 for x in points_orig]

point_centers = [0.0016793018999999999, 0.00067773325333333, 0.0006662270949999999, 0.00039591994000000004, 0.000230577926]
point_errors =  [0.0006452832008827938, 6.5981889313042E-5 , 0.00013455174063625334, 7.631636676443212e-05, 7.6693627029069717e-05]

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
plt.ylabel('Classification loss', fontsize=16)
plt.title('Classification loss from AL training phase 1 and 3 and bulk training', fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.show()
plt.savefig("classification-loss-AL-vs-bulk")
plt.close()
