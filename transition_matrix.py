import pandas as pd
import numpy as np


snakes = {16:6,47:26,49:11,56:53, 62:19, 64:60, 87:24,93:73,95:75,98:78}
snake_keys = list(snakes.keys())
ladders = {1:38,4:14,9:31,21:42,28:84,36:44,51:67,71:91,80:100}
ladder_keys = list(ladders.keys())
end_point = 100
N = 101
matrix = np.zeros((N, N))

for i in range(N):
    if i == 100:
        matrix[i, i] = 1  
        continue

    for roll in range(1, 7):
        target = i + roll
        if target > 100:
            target = 100
        if target in snakes:
            target = snakes[target]
        elif target in ladders:
            target = ladders[target]

        matrix[i, target] += 1/6

df = pd.DataFrame(matrix, columns=range(101),index=range(101))
print(df.shape)
df.to_csv(r"c:\Users\anime\Downloads\Snakes and Ladders Transition Matrix.csv")
