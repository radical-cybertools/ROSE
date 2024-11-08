import matplotlib.pyplot as plt

#It should be 1, 2 or 3
AL_phase_idx=3
# Data setup
centers = [0.00147374432,           0.000324869774,              3.58019838e-05]  # Central values of each band
errors =  [0.00038760632010833565,  0.00011294765746099644,      7.636654988751802e-06]  # Errors or half-widths of each band

center = centers[AL_phase_idx-1]
error = errors[AL_phase_idx-1]

# Additional data for vertical bars
points =        [4500,                  18000,                 36000, 54000, 72000]
point_centers = [0.00634084114,         0.0005015651040000001, 2.69187898e-05, 1.1652123e-06, 5.017965216666667e-07]  # Y positions for each point
point_errors =  [0.0008063084784575381, 9.620696886758185e-05, 8.981106109070761e-06, 1.9382898728489505e-07, 4.9801772917032595e-08]  # Errors for each point

# Plot setup
plt.figure(figsize=(12, 8))

plt.yscale('log')

plt.hlines(y=center + error, xmin=0, xmax=80000, color='red', linewidth=2, label="Error band from training with AL")
plt.hlines(y=center - error, xmin=0, xmax=80000, color='red', linewidth=2)

for x, y, e in zip(points, point_centers, point_errors):
    plt.errorbar(x, y, yerr=e, fmt='o', color='black', capsize=5, label="Error bar from bulk training")

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # Creating a dictionary to remove duplicates
plt.legend(by_label.values(), by_label.keys())

# Customization
#plt.ylim(0, 0.0025)
plt.xlim(0, 80000)
plt.xlabel('Number of samples')
plt.ylabel('Average Uncertainty')
plt.title('Average uncertainty from AL training phase-{} ({} samples in total) and bulk training with different number of samples'.format(AL_phase_idx, int((AL_phase_idx + 1) * 4500)))

# Show the plot
plt.show()
plt.savefig("Average-uncertainty-AL-phase-{}".format(AL_phase_idx))
plt.close()
