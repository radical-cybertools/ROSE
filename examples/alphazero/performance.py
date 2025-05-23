import matplotlib.pyplot as plt
import pickle

# Configuration
LOG_PATH = 'logs/performance_log.pkl'
PLOT_PATH = 'logs/performance_plot.png'

# Load performance log
with open(LOG_PATH, 'rb') as f:
    performance_log = pickle.load(f)

# Extract metrics
episodes = list(range(len(performance_log)))
win_rates = [entry['win_rate'] for entry in performance_log]

# Plot performance
plt.figure(figsize=(10, 5))
plt.plot(episodes, win_rates, label='Win Rate')
plt.xlabel('Episode')
plt.ylabel('Win Rate')
plt.title('Performance Over Time')
plt.legend()
plt.savefig(PLOT_PATH)
plt.close()
