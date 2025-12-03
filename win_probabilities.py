import pandas as pd
import numpy as np


P = pd.read_csv(r"C:\Users\anime\Downloads\Snakes and Ladders Transition Matrix.csv")
P = P.drop('Unnamed: 0', axis =1)


P = P.to_numpy()

#print(P.shape)
win_times = np.arange(0,251)
win_probs = []
for n in range(0,251):


    win_probs.append(np.linalg.matrix_power(P, n)[0,100])

win_probs_pdf = []
win_probs_pdf.append(win_probs[0])

for x in range(1,251):

    win_probs_pdf.append(win_probs[x]-win_probs[x-1])



import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))

plt.bar(win_times, win_probs_pdf, color = 'red')



plt.xlabel('Winning Time')
plt.ylabel('Probability Density')
plt.title('Plot of Probability Density Function for winning times')
plt.show()



X,Y = np.meshgrid(win_times, win_times, indexing='ij')
PX, PY = np.meshgrid(win_probs_pdf, win_probs_pdf, indexing='ij')

joint_probs = PX*PY

p_X_less_Y = np.sum(joint_probs[X < Y])
p_X_equal_Y = np.sum(joint_probs[X == Y])
p_X_greater_Y = np.sum(joint_probs[X > Y])

values = [p_X_less_Y + p_X_equal_Y, p_X_greater_Y]
categories = ['Player 1 Wins', 'Player 2 Wins']
plt.bar(categories,values, color = 'blue')
plt.title('Win Probabilities for Two Player Game')
plt.xlabel('Player')
plt.ylabel('Win Probability')
plt.show()

print(values)
print(p_X_equal_Y)
print(p_X_less_Y -p_X_greater_Y)


def hitting_time(P, start_state, absorbing_state):
    state = start_state
    steps = 0

    while state != absorbing_state:
        state = np.random.choice(len(P), p=P[state])
        steps += 1

    return steps

player_1_record = []
for x in range(0,10000):

    #print(x)

    p1 = hitting_time(P,0,100)
    p2 = hitting_time(P,0,100)

    if p1 <= p2:

        player_1_record.append(1)

    else:
        player_1_record.append(0)

sum = 0
for x in player_1_record:

    sum += x
values = [sum/10000, (10000-sum)/10000]

plt.bar(categories,values, color = 'red')
plt.title('Win Probabilities for Two Player Game (Simulated Averages n=10k)')
plt.xlabel('Player')
plt.ylabel('Win Probability')
plt.show()

print(values)

import seaborn as sns
#plt.title('Heatmap Showing Joint PDF of Player 1 and Player 2 Winning Times')
#sns.heatmap(joint_probs)

#plt.show()


X,Y, Z = np.meshgrid(win_times, win_times, win_times, indexing='ij')
PX, PY, PZ = np.meshgrid(win_probs_pdf, win_probs_pdf,win_probs_pdf, indexing='ij')

joint_probs = PX*PY*PZ

p_X_less_Y = np.sum(joint_probs[X < Y])
p_X_equal_Y = np.sum(joint_probs[X == Y])
p_X_less_Z = np.sum(joint_probs[X < Z])
p_X_equal_Z = np.sum(joint_probs[X == Z])
p_Y_less_Z = np.sum(joint_probs[Y < Z])
p_Y_equal_Z = np.sum(joint_probs[Y == Z])
p_X_greater_Y = np.sum(joint_probs[X > Y])
p_Z_less_Y = np.sum(joint_probs[Z < Y])
p_Z_less_X = np.sum(joint_probs[Z < X])
p1 = np.sum(joint_probs[(X <= Y) & (X <= Z)])        # X is min (ties allowed)
p2 = np.sum(joint_probs[(Y < X) & (Y <= Z)])         # Y is min, strictly < X
p3 = np.sum(joint_probs[(Z < X) & (Z < Y)])
values = [p1,p2,p3]
categories = ['Player 1 Wins', 'Player 2 Wins', 'Player 3 Wins']
plt.bar(categories,values, color = 'blue')
plt.title('Win Probabilities for Three Player Game')
plt.xlabel('Player')
plt.ylabel('Win Probability')
plt.show()

print(values)
print(p_X_equal_Y)
print(p_X_less_Y -p_X_greater_Y)


def hitting_time(P, start_state, absorbing_state):
    state = start_state
    steps = 0

    while state != absorbing_state:
        state = np.random.choice(len(P), p=P[state])
        steps += 1

    return steps

player_1_record = []
player_2_record = []
player_3_record = []
for x in range(0,10000):

    #print(x)

    p1 = hitting_time(P,0,100)
    p2 = hitting_time(P,0,100)
    p3 = hitting_time(P,0,100)

    if p1 <= p2 and p1 <= p3:

        player_1_record.append(1)
        player_2_record.append(0)
        player_3_record.append(0)

    elif p2 <= p3 and p2 < p1:
        player_2_record.append(1)
        player_1_record.append(0)
        player_3_record.append(0)

    elif p3 < p1 and p3 < p2:
        player_2_record.append(0)
        player_1_record.append(0)
        player_3_record.append(1)


    
    


player_1_record = np.array(player_1_record)
player_2_record = np.array(player_2_record)
player_3_record = np.array(player_3_record)

values = [player_1_record.sum()/10000, player_2_record.sum()/10000, player_3_record.sum()/10000]
plt.bar(categories,values, color = 'red')
plt.title('Win Probabilities for Three Player Game (Simulated Averages n=10k)')
plt.xlabel('Player')
plt.ylabel('Win Probability')
plt.show()

print(values)


def compute_player_win_prob(win_probs, win_probs_pdf, player_index, total_players):
    probability = 0
    for x in range(0, len(win_probs_pdf)):

        if x > 0:
            probability += win_probs_pdf[x] * (1-win_probs[x])**(player_index-1) * (1-win_probs[x-1])**(total_players-player_index)

    return probability


    

      

# Probabilities
p1 = compute_player_win_prob(win_probs, win_probs_pdf, 1, 4)
p2 = compute_player_win_prob(win_probs, win_probs_pdf, 2, 4)
p3 = compute_player_win_prob(win_probs, win_probs_pdf, 3, 4)
p4 = compute_player_win_prob(win_probs, win_probs_pdf, 4, 4)
values = [p1,p2,p3, p4]
categories = ['Player 1 Wins', 'Player 2 Wins', 'Player 3 Wins', 'Player 4 Wins']
plt.bar(categories,values, color = 'blue')
plt.title('Win Probabilities for Four Player Game')
plt.xlabel('Player')
plt.ylabel('Win Probability')
plt.show()

print(values)
print(p_X_equal_Y)
print(p_X_less_Y -p_X_greater_Y)


def hitting_time(P, start_state, absorbing_state):
    state = start_state
    steps = 0

    while state != absorbing_state:
        state = np.random.choice(len(P), p=P[state])
        steps += 1

    return steps

player_1_record = []
player_2_record = []
player_3_record = []
player_4_record = []
for x in range(0,10000):

    #print(x)

    p1 = hitting_time(P,0,100)
    p2 = hitting_time(P,0,100)
    p3 = hitting_time(P,0,100)
    p4 = hitting_time(P, 0,100)

    if p1 <= p2 and p1 <= p3 and p1 <= p4:

        player_1_record.append(1)
        player_2_record.append(0)
        player_3_record.append(0)
        player_4_record.append(0)

    elif p2 <= p3 and p2 <= p4 and p2 < p1:
        player_2_record.append(1)
        player_1_record.append(0)
        player_3_record.append(0)
        player_4_record.append(0)

    elif p3 < p1 and p3 < p2 and p3 <= p4:
        player_2_record.append(0)
        player_1_record.append(0)
        player_3_record.append(1)
        player_4_record.append(0)

    elif p4 < p1 and p4 < p2 and p4 < p3:
        player_2_record.append(0)
        player_1_record.append(0)
        player_3_record.append(0)
        player_4_record.append(1)


    
    


player_1_record = np.array(player_1_record)
player_2_record = np.array(player_2_record)
player_3_record = np.array(player_3_record)
player_4_record = np.array(player_4_record)
values = [player_1_record.sum()/10000, player_2_record.sum()/10000, player_3_record.sum()/10000,player_4_record.sum()/10000]
plt.bar(categories,values, color = 'red')
plt.title('Win Probabilities for Four Player Game (Simulated Averages n=10k)')
plt.xlabel('Player')
plt.ylabel('Win Probability')
plt.show()

print(values)



def compute_player_win_prob(win_probs, win_probs_pdf, player_index, total_players):
    probability = 0
    for x in range(0, len(win_probs_pdf)):

        if x > 0:
            probability += win_probs_pdf[x] * (1-win_probs[x])**(player_index-1) * (1-win_probs[x-1])**(total_players-player_index)

    return probability


    

      

# Probabilities
p1 = compute_player_win_prob(win_probs, win_probs_pdf, 1, 5)
p2 = compute_player_win_prob(win_probs, win_probs_pdf, 2, 5)
p3 = compute_player_win_prob(win_probs, win_probs_pdf, 3, 5)
p4 = compute_player_win_prob(win_probs, win_probs_pdf, 4, 5)
p5 = compute_player_win_prob(win_probs, win_probs_pdf, 5, 5)
values = [p1,p2,p3, p4, p5]
categories = ['Player 1 Wins', 'Player 2 Wins', 'Player 3 Wins', 'Player 4 Wins', 'Player 5 Wins']
plt.bar(categories,values, color = 'blue')
plt.title('Win Probabilities for Five Player Game')
plt.xlabel('Player')
plt.ylabel('Win Probability')
plt.show()

print(values)
print(p_X_equal_Y)
print(p_X_less_Y -p_X_greater_Y)


def hitting_time(P, start_state, absorbing_state):
    state = start_state
    steps = 0

    while state != absorbing_state:
        state = np.random.choice(len(P), p=P[state])
        steps += 1

    return steps

player_1_record = []
player_2_record = []
player_3_record = []
player_4_record = []
player_5_record = []
for x in range(0,10000):

    #print(x)

    p1 = hitting_time(P,0,100)
    p2 = hitting_time(P,0,100)
    p3 = hitting_time(P,0,100)
    p4 = hitting_time(P, 0,100)
    p5 = hitting_time(P, 0, 100)

    if p1 <= p2 and p1 <= p3 and p1 <= p4 and p1 <= p5:

        player_1_record.append(1)
        player_2_record.append(0)
        player_3_record.append(0)
        player_4_record.append(0)
        player_5_record.append(0)

    elif p2 <= p3 and p2 <= p4 and p2 < p1 and p2<=p5:
        player_2_record.append(1)
        player_1_record.append(0)
        player_3_record.append(0)
        player_4_record.append(0)
        player_5_record.append(0)

    elif p3 < p1 and p3 < p2 and p3 <= p4 and p3 <= p5:
        player_2_record.append(0)
        player_1_record.append(0)
        player_3_record.append(1)
        player_4_record.append(0)
        player_5_record.append(0)

    elif p4 < p1 and p4 < p2 and p4 < p3 and p4 <= p5:
        player_2_record.append(0)
        player_1_record.append(0)
        player_3_record.append(0)
        player_4_record.append(1)
        player_5_record.append(0)
    
    else:
        player_2_record.append(0)
        player_1_record.append(0)
        player_3_record.append(0)
        player_4_record.append(0)
        player_5_record.append(1)




    
    


player_1_record = np.array(player_1_record)
player_2_record = np.array(player_2_record)
player_3_record = np.array(player_3_record)
player_4_record = np.array(player_4_record)
player_5_record = np.array(player_5_record)
values = [player_1_record.sum()/10000, player_2_record.sum()/10000, player_3_record.sum()/10000,player_4_record.sum()/10000,player_5_record.sum()/10000]
plt.bar(categories,values, color = 'red')
plt.title('Win Probabilities for Five Player Game (Simulated Averages n=10k)')
plt.xlabel('Player')
plt.ylabel('Win Probability')
plt.show()

print(values)



x = np.arange(1,50)
player_1_vs_baseline = []
player_k_vs_baseline = []
standard_deviation_vs_baseline = []
player_1_vs_player_k = []
average_risk_ratio_for_player_1 = []
average_risk_ratio_for_player_k = []

for i in range(1,50):
    
    player_1_vs_baseline.append(compute_player_win_prob(win_probs, win_probs_pdf, 1, i)/(1/i)) 
    player_k_vs_baseline.append(compute_player_win_prob(win_probs, win_probs_pdf, i, i)/(1/i))
    player_1_vs_player_k.append(compute_player_win_prob(win_probs, win_probs_pdf, 1, i)/compute_player_win_prob(win_probs, win_probs_pdf, i, i))
    
    player_1_risk_ratios = []
    player_k_risk_ratios = []

    for b in range(1, i+1):

        player_1_risk_ratios.append(compute_player_win_prob(win_probs, win_probs_pdf, 1, i)/compute_player_win_prob(win_probs, win_probs_pdf, b, i))
        player_k_risk_ratios.append(compute_player_win_prob(win_probs, win_probs_pdf, i, i)/compute_player_win_prob(win_probs, win_probs_pdf, b, i))

    player_1_risk_ratios = np.mean(np.array(player_1_risk_ratios))
    player_k_risk_ratios = np.mean(np.array(player_k_risk_ratios))

    average_risk_ratio_for_player_1.append(player_1_risk_ratios)
    average_risk_ratio_for_player_k.append(player_k_risk_ratios)


    risks = []

    for a in range(1,i+1):

        risks.append(compute_player_win_prob(win_probs, win_probs_pdf, a, i)/(1/i))

    risks = np.array(risks)

    standard_deviation_vs_baseline.append(np.std(risks, ddof=0))







plt.figure(figsize=(10,10))
plt.plot(x,player_1_vs_baseline,label='Player 1 Risk Ratio (Baseline Comparison)', color='red')
plt.plot(x,player_k_vs_baseline,label='Player K Risk Ratio (Baseline Comparison)', color = 'blue')
plt.plot(x,standard_deviation_vs_baseline,label='Standard Deviation of Risk Ratios (Baseline Comparison)', color = 'orange')
plt.plot(x,player_1_vs_player_k,label='Player 1 vs Player K Risk Ratio', color = 'purple')
plt.plot(x,average_risk_ratio_for_player_1,label='Average Risk Ratio for Player 1 vs. All Players', color = 'green')
plt.plot(x,average_risk_ratio_for_player_k,label='Average Risk Ratio for Player K vs. All Players', color = 'black')
plt.title('Analysis of Game Fairness as Number of Players Increase')
plt.xlabel('Number of Players')
plt.ylabel('Risk Ratio')
plt.legend()
plt.show()


