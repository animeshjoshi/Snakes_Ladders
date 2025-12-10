import numpy as np
snakes = {16:6,47:26,49:11,56:53, 62:19, 64:60, 87:24,93:73,95:75,98:78}
snake_dis = list(snakes.keys())
ladders = {1:38,4:14,9:31,21:42,28:84,36:44,51:67,71:91,80:100}
ladder_dis = list(ladders.keys())
snakes = np.load('snakes.npy')[1]- np.load('baseline.npy')[1]
ladders = np.load('ladders.npy')[1] - np.load('baseline.npy')[1]
#snake_dis = [10, 21, 38, 3, 43, 4, 63, 20, 20, 20]
#ladder_dis = [37, 10, 22, 21, 56, 8, 16, 20, 20]


import matplotlib.pyplot as plt

# Example arrays (replace with your actual data)

# Create the plot
plt.scatter(snake_dis, snakes, color='blue', label='Snake Position vs. Change in Average Risk Ratio for P1')  # Blue line
plt.scatter(ladder_dis, ladders, color='red', label='Ladder Position vs. Change in Average Risk Ratio for P1')   # Red line
# Regression line for first scatter
x1 = snake_dis
y1 = snakes
x2 = ladder_dis
y2 = ladders
coef1 = np.polyfit(x1, y1, 1)
coef2 = np.polyfit(x2, y2, 1)

# Create extended x-range for the lines
x_line = np.linspace(min(min(x1), min(x2)) - 1, max(max(x1), max(x2)) + 1, 100)

# Compute y-values for regression lines
y1_fit = np.polyval(coef1, x_line)
y2_fit = np.polyval(coef2, x_line)
plt.text(x_line[0], y1_fit[0]+0.5, f'y = {coef1[0]:.2f}x + {coef1[1]:.2f}', color='blue')
plt.text(x_line[0], y2_fit[0]-1, f'y = {coef2[0]:.2f}x + {coef2[1]:.2f}', color='red')

# Plot regression lines
plt.plot(x_line, y1_fit, color='blue')
plt.plot(x_line, y2_fit, color='red')
plt.title('Snake/Ladder Positions and Removal Effects on Player 1 Average Advantage')
plt.xlabel('Snake/Ladder Positions')
plt.ylabel('Change in Risk Ratio')

# Add legend
plt.legend()

# Show the plot
plt.show()
