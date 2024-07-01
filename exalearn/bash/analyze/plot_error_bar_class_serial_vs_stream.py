import matplotlib.pyplot as plt

central_xs = [0, 1, 2, 3]

serial_xs = [x-0.02 for x in central_xs]
serial_centers = [0.0016793018999999999, 0.000407634226,          0.00018441775199999998,     7.3676936e-05]
serial_errors =  [0.0006452832008827938, 0.00015972274060767233,  6.452705198370165e-05,      5.934087042221745e-06]
serial_samples = [13500, 27000, 40500, 54000]

stream_xs = [x+0.02 for x in central_xs]
stream_centers = [0.00168246485, 0.000571516245, 0.0002772049083333333, 5.6518853666666676e-05]
stream_errors =  [0.00120053502198615, 0.0002986175203921493, 7.570235156133183e-05, 1.7664624873785896e-05]
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
plt.ylabel('Classification loss', fontsize=16)
plt.title('Classification loss from AL training with serial and streaming execution', fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.show()
plt.savefig("classification-loss-serial-vs-stream")
plt.close()
