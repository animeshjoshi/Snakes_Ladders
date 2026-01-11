import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = r"C:\Users\Terre\Desktop\文件\py文件\Snakes and Ladders Transition Matrix.csv"
df_raw = pd.read_csv(path, header=None)


P = df_raw.iloc[1:, 1:].to_numpy(dtype=float)
print("P shape:", P.shape)         


n_states = P.shape[0]      
n_transient = n_states - 1 

A = np.zeros((n_transient, n_transient))
b = np.ones(n_transient)   

for i in range(n_transient):       
    A[i, i] = 1.0
    for j in range(n_transient):   
        A[i, j] -= P[i, j]         

E = np.linalg.solve(A, b)
E0 = E[0]


plt.figure(figsize=(10, 5))
plt.plot(range(n_transient), E)  
plt.xlabel("Square i")
plt.ylabel("Expected rolls remaining  E(i)")
plt.title("Expected Rolls Remaining from Each Square (E(i))")
plt.grid(True)
plt.tight_layout()
plt.show()
print("Figure saved as E_i_curve.png")
