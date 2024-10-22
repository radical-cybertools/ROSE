import matplotlib.pyplot as plt

central_xs = [0, 1, 2, 3]

serial_xs = [x-0.02 for x in central_xs]
serial_centers = [0.00107867866, 0.000190619708,          9.01861632e-05,       4.96659404e-05]
serial_errors =  [0.0003555528552316384, 5.559355422743267e-05,   3.22970222452586e-05, 1.3117838937239361e-05]
serial_samples = [13500, 27000, 40500, 54000]

stream_xs = [x+0.02 for x in central_xs]
stream_centers = [0.001580492033333333, 0.0002630935366666667, 0.00016263416333333334, 4.849238966666667e-05]
stream_errors =  [0.0005568699656732341, 7.723198931743646e-05, 3.4741286017265134e-05, 3.7501536159783587e-06]
stream_samples = [13500, 21600, 37800, 59400]

# Plot setup
plt.figure(figsize=(12, 8))
plt.yscale('log')

for x, y, e in zip(serial_xs, serial_centers, serial_errors):
    plt.errorbar(x, y, yerr=e, fmt='o', color='black', capsize=5, label="Error bar from serial execution")
for x, y, e in zip(stream_xs, stream_centers, stream_errors):
    plt.errorbar(x, y, yerr=e, fmt='o', color='red', capsize=5, label="Error bar from streaming execution")
for i, (x, y) in enumerate(zip(stream_xs, stream_centers)):
    plt.text(x + 0.02, y*1.1, "{} samples".format(serial_samples[i]), verticalalignment='center', color='black', fontsize=16)
    plt.text(x + 0.02, y*0.9, "{} samples".format(stream_samples[i]), verticalalignment='center', color='red', fontsize=16)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # Creating a dictionary to remove duplicates
plt.legend(by_label.values(), by_label.keys(), fontsize=16)

# Customization
#plt.ylim(0, 0.0025)
plt.xlim(-0.2, 3.7)
plt.xlabel('AL phase', fontsize=16)
plt.ylabel('l2 difference', fontsize=16)
plt.title('l2 difference from AL training with serial and streaming execution', fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.show()
plt.savefig("l2-difference-serial-vs-stream")
plt.close()
