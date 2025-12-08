import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\Terre\Desktop\文件\py文件\Snakes and Ladders Transition Matrix.csv", header=None)
P = df.iloc[1:, 1:].to_numpy(dtype=float)   
N = 101   

E = np.zeros(N)           
tol = 1e-10                
max_iter = 20000           

for k in range(max_iter):
    newE = np.zeros_like(E)
    newE[100] = 0          


    for i in range(100):
        newE[i] = 1 + np.dot(P[i], E)

    diff = np.max(np.abs(newE - E))
    if diff < tol:
        E = newE
        break

    E = newE

print(f"\nExpected rolls from start state E(0) = {E[0]:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(range(N), E)
plt.xlabel("Square i")
plt.ylabel("Expected rolls remaining  E(i)")
plt.title("Expected Rolls Remaining from Each Square (E(i))")
plt.grid(True)
plt.tight_layout()

plt.savefig("E_i_curve.png", dpi=300)
print("\nFigure saved as E_i_curve.png")

plt.show()

