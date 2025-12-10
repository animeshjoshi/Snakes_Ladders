import pandas as pd
import numpy as np


snakes = {16:6,47:26,49:11,56:53, 62:19, 64:60, 87:24,93:73,95:75,98:78}
snake_keys = list(snakes.keys())
ladders = {1:38,4:14,9:31,21:42,28:84,36:44,51:67,71:91,80:100}
ladder_keys = list(ladders.keys())
end_point = 100
N = 101
matrix = np.zeros((N, N))



def compute_player_win_prob(win_probs, win_probs_pdf, player_index, total_players):
    probability = 0
    for x in range(0, len(win_probs_pdf)):

        if x > 0:
            probability += win_probs_pdf[x] * (1-win_probs[x])**(player_index-1) * (1-win_probs[x-1])**(total_players-player_index)

    return probability



player_combos = [2,5,10,15]
snake_array = []
ladder_array = []
for num in player_combos:

    ladder_risk_ratios = []
    for key in list(ladders.keys()):  # list() so we can safely modify the dict
            value = ladders.pop(key)     
            
            print(f"Popped {key}: {value}")

            matrix = np.zeros((N, N))
            snake_keys = list(snakes.keys())
            ladder_keys = list(ladders.keys())

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

            P = pd.read_csv(r"C:\Users\anime\Downloads\Snakes and Ladders Transition Matrix.csv")
            P = P.drop('Unnamed: 0', axis =1)


            P = P.to_numpy()

            win_times = np.arange(0,251)
            win_probs = []
            for n in range(0,251):


                win_probs.append(np.linalg.matrix_power(P, n)[0,100])

            win_probs_pdf = []
            win_probs_pdf.append(win_probs[0])

            for x in range(1,251):

                win_probs_pdf.append(win_probs[x]-win_probs[x-1])

            
            player_1_risk_ratios = []
            player_k_risk_ratios = []

            for b in range(1, num+1):

                player_1_risk_ratios.append(compute_player_win_prob(win_probs, win_probs_pdf, 1, num)/compute_player_win_prob(win_probs, win_probs_pdf, b, num))
                #player_k_risk_ratios.append(compute_player_win_prob(win_probs, win_probs_pdf, i, i)/compute_player_win_prob(win_probs, win_probs_pdf, b, i))

            player_1_risk_ratios = np.mean(np.array(player_1_risk_ratios))
            #player_k_risk_ratios = np.mean(np.array(player_k_risk_ratios))

            ladder_risk_ratios.append(player_1_risk_ratios)
     



            
            
            ladders[key] = value
            
            
           
    snake_risk_ratios = []
    for key in list(snakes.keys()):  # list() so we can safely modify the dict
            value = snakes.pop(key)     
            
            print(f"Popped {key}: {value}")

            matrix = np.zeros((N, N))
            snake_keys = list(snakes.keys())
            ladder_keys = list(ladders.keys())

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

            P = pd.read_csv(r"C:\Users\anime\Downloads\Snakes and Ladders Transition Matrix.csv")
            P = P.drop('Unnamed: 0', axis =1)


            P = P.to_numpy()

            win_times = np.arange(0,251)
            win_probs = []
            for n in range(0,251):


                win_probs.append(np.linalg.matrix_power(P, n)[0,100])

            win_probs_pdf = []
            win_probs_pdf.append(win_probs[0])

            for x in range(1,251):

                win_probs_pdf.append(win_probs[x]-win_probs[x-1])

            
            player_1_risk_ratios = []
            player_k_risk_ratios = []

            for b in range(1, num+1):

                player_1_risk_ratios.append(compute_player_win_prob(win_probs, win_probs_pdf, 1, num)/compute_player_win_prob(win_probs, win_probs_pdf, b, num))
                #player_k_risk_ratios.append(compute_player_win_prob(win_probs, win_probs_pdf, i, i)/compute_player_win_prob(win_probs, win_probs_pdf, b, i))

            player_1_risk_ratios = np.mean(np.array(player_1_risk_ratios))
            #player_k_risk_ratios = np.mean(np.array(player_k_risk_ratios))

            snake_risk_ratios.append(player_1_risk_ratios)
     



            
            
            snakes[key] = value

    snake_array.append(snake_risk_ratios)
    ladder_array.append(ladder_risk_ratios)

baseline_array = []
matrix = np.zeros((N, N))
snake_keys = list(snakes.keys())
ladder_keys = list(ladders.keys())

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

P = pd.read_csv(r"C:\Users\anime\Downloads\Snakes and Ladders Transition Matrix.csv")
P = P.drop('Unnamed: 0', axis =1)


P = P.to_numpy()

win_times = np.arange(0,251)
win_probs = []
for n in range(0,251):


    win_probs.append(np.linalg.matrix_power(P, n)[0,100])

win_probs_pdf = []
win_probs_pdf.append(win_probs[0])
for x in range(1,251):

    win_probs_pdf.append(win_probs[x]-win_probs[x-1])

            

            
for x in player_combos:
    player_1_risk_ratios = []
    for b in range(1, x+1):

        player_1_risk_ratios.append(compute_player_win_prob(win_probs, win_probs_pdf, 1, x)/compute_player_win_prob(win_probs, win_probs_pdf, b, x))
                    #player_k_risk_ratios.append(compute_player_win_prob(win_probs, win_probs_pdf, i, i)/compute_player_win_prob(win_probs, win_probs_pdf, b, i))

    player_1_risk_ratios = np.mean(np.array(player_1_risk_ratios))
                #player_k_risk_ratios = np.mean(np.array(player_k_risk_ratios))

    baseline_array.append(player_1_risk_ratios)


print(snake_array)
print(ladder_array)
print(baseline_array)
snake_array = np.array(snake_array)
np.save('snakes.npy', snake_array)
ladder_array = np.array(ladder_array)
np.save('ladders.npy', ladder_array)
baseline_array = np.array(baseline_array)
np.save('baseline.npy', baseline_array)


