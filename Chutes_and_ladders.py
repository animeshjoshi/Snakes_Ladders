import numpy as np

MATRIX = None


def initialize_matrix(file_path):
    """
    Loads the matrix from the CSV file.
    """
    global MATRIX
    MATRIX = np.loadtxt(file_path, delimiter=",")


def matrix_power(n):
    """
    Computes MATRIX^n.
    """
    if MATRIX is None:
        raise ValueError("Matrix not initialized. Call initialize_matrix() first.")
    return np.linalg.matrix_power(MATRIX, n+1)


def get_value(n, row, col):
    """
    Returns the value at (row, col) after raising MATRIX to the nth power.
    Row and col are 0-indexed.
    """
    powered = matrix_power(n)
    return powered[row, col]

initialize_matrix("C:/Users/elain/Downloads/chutes and ladders.csv")

# getting player names
player1 = input("Who is player 1? ")
player2 = input("who is player 2? ")

# starting the loop for each turn to input the current states until someone wins
go = True
while go:
    nextPlayer = input("Who is going next? ")
    while nextPlayer != player1 and nextPlayer != player2:
        print("Please type a player's name...")
        nextPlayer = input("Who is going next? ")

    # find where A and B are, where A is the player that goes next.
    if nextPlayer == player1:
        A = input(str("Where is " + player1 + "? "))
        B = input(str("Where is " + player2 + "? "))
    else:
        A = input(str("Where is " + player2 + "? "))
        B = input(str("Where is " + player1 + "? "))

    WIN = 100   # absorbing index

    is_A_valid = A.isnumeric() and 0 <= int(A) <= 100
    is_B_valid = B.isnumeric() and 0 <= int(B) <= 100

    if not (is_A_valid & is_B_valid):
        print("non-valid inputs.")
    else:
        A = int(A)
        B = int(B)
        if A == 100 or B == 100:
            go = False

        print(" ")

        # starting the loop of calculating for each turn until we know what rounded to 3 decimals must be
        SumA = 0
        SumB = 0
        PA_prev = 0
        PB_prev = 0
        x = 0
        Sum = 0
        while Sum != 1:
            # starting on the first turn x
            x += 1

            # Prob. wins on or before turn x
            PA_by = get_value(x, A, WIN)
            PB_by = get_value(x, B, WIN)

            # Prob. wins on turn x
            PA_on = PA_by - PA_prev
            PB_on = PB_by - PB_prev

            # Prob. A wins on this turn AND B has not already won
            SumA += PA_on * (1 - PB_prev)

            # Prob. B wins on this turn AND A has not done so on the same # of turn
            SumB += PB_on * (1 - PA_by)

            PA_prev = PA_by
            PB_prev = PB_by

            Sum = round(SumA, 3) + round(SumB, 3)

            print("turn", x, "| A:", f"{SumA:.2f}", "| B:", f"{SumB:.2f}", "| Sure-ness:", Sum)

        print(" ")
        if nextPlayer == player1:
            # then A is for player1, and B is player2
            print(player1, "'s probability of winning is", round(SumA, 3))
            print(player2, "'s probability of winning is", round(SumB, 3))
        else:
            # then B is for player1, and A is player1
            print(player1, "'s probability of winning is", round(SumB, 3))
            print(player2, "'s probability of winning is", round(SumA, 3))
        print("calculated up to", x + x - 1, "turns, or until certain to round to the above values by 3 decimal places.")

    print(" ")
