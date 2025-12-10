import tkinter as tk
import numpy as np
import random as random

BOARD_SIZE = 10
CELL = 50

# Chutes and ladders
LADDERS = {1: 38, 4: 14, 9: 31, 21: 42, 28: 84, 36: 44, 51: 67, 71: 91, 80: 100}
CHUTES = {98: 78, 95: 75, 93: 73, 87: 24, 64: 60, 62: 19, 56: 53, 49: 11, 47: 26, 16: 6}

# preset moves for turn-by-turn play
PRESET = [2, 3, 2, 3, 3, 5, 5, 6, 3, 5, 3, 5, 1, 2, 2]

MATRIX = None


def initialize_matrix(file_path):
    global MATRIX
    MATRIX = np.loadtxt(file_path, delimiter=",")


def matrix_power(n):
    if MATRIX is None:
        raise ValueError("Matrix not initialized. Call initialize_matrix() first.")
    return np.linalg.matrix_power(MATRIX, n+1)


def get_value(n, row, col):
    powered = matrix_power(n)
    return powered[row, col]

initialize_matrix("C:/Users/elain/Downloads/chutes and ladders.csv")


def compute_win_probabilities(player1, player2, turn):
    # A represents the player whos turn it is.
    if turn == 1:
        A = player1
        B = player2
    else:
        A = player2
        B = player1

    # starting the loop of calculating for each turn until we know what rounded to 3 decimals must be
    SumA = 0
    SumB = 0
    PA_prev = 0
    PB_prev = 0
    x = 0
    Sum = 0
    WIN = 100
    while Sum != 1:
        # starting on the first turn x
        x += 1

        # Prob. wins on or before turn x
        PA_by = get_value(x, A, WIN)
        PB_by = get_value(x, B, WIN)

        # Prob. wins on turn x
        PA_on = PA_by - PA_prev
        PB_on = PB_by - PB_prev

        # Prob. A wins on x AND B has not won by x
        SumA += PA_on * (1 - PB_prev)

        # Prob. B wins on x AND A has not won on or by x
        SumB += PB_on * (1 - PA_by)

        PA_prev = PA_by
        PB_prev = PB_by

        Sum = round(SumA, 3) + round(SumB, 3)

    print(" ")
    if turn == 1:
        # then A is for player1, and B is player2
        p1_win = round(SumA, 3)
        p2_win = round(SumB, 3)
    else:
        # then B is for player1, and A is player1
        p1_win = round(SumB, 3)
        p2_win = round(SumA, 3)

    return p1_win, p2_win


class ChutesLaddersDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("Chutes and Ladders Demo")

        self.prob_top = tk.Label(root, text="Player 1 (yellow) win prob: --%", font=("Arial", 14))
        self.prob_top.pack(pady=5)

        self.canvas = tk.Canvas(root, width=BOARD_SIZE * CELL, height=BOARD_SIZE * CELL, bg="white")
        self.canvas.pack()

        self.draw_board()

        self.p1_pos = 0
        self.p2_pos = 0

        self.p1_token = self.canvas.create_oval(0, 0, 0, 0, fill="yellow")
        self.p2_token = self.canvas.create_oval(0, 0, 0, 0, fill="purple")
        self.update_player_tokens()

        self.prob_bottom = tk.Label(root, text="Player 2 (purple) win prob: --%", font=("Arial", 14))
        self.prob_bottom.pack(pady=5)

        self.move_index = 0
        self.current_player = 1  # 1 or 2

        self.btn = tk.Button(root, text="Next Move", command=self.next_move)
        self.btn.pack(pady=10)

        self.update_probabilities()

        self.move_arrow = None  # store the canvas arrow so we can delete it

    def draw_board(self):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                x = c * CELL
                y = (BOARD_SIZE - 1 - r) * CELL
                self.canvas.create_rectangle(x, y, x+CELL, y+CELL, outline="black")

        # Draw ladders (green) and chutes (brown)
        for start, end in LADDERS.items():
            self.draw_arrow(start, end, "green")
        for start, end in CHUTES.items():
            self.draw_arrow(start, end, "red")

    def square_center(self, sq):
        if sq == 0:
            # Place off-board start area
            x, y = self.square_center(1)
            return x, y + CELL  # drop below the board

        sq -= 1  # accounting for index

        r = sq // BOARD_SIZE
        c = sq % BOARD_SIZE

        if r % 2 == 1:
            c = BOARD_SIZE - 1 - c

        x = c * CELL + CELL / 2
        y = (BOARD_SIZE - 1 - r) * CELL + CELL / 2
        return x, y

    def draw_arrow(self, start, end, color):
        x1, y1 = self.square_center(start)
        x2, y2 = self.square_center(end)
        self.canvas.create_line(x1, y1, x2, y2, arrow=tk.LAST, width=3, fill=color)

    def update_player_tokens(self):
        self.place_token(self.p1_token, self.p1_pos, -10)
        self.place_token(self.p2_token, self.p2_pos, +10)

    def place_token(self, token_id, pos, y_offset):
        x, y = self.square_center(pos)
        r = 12
        self.canvas.coords(token_id, x-r, y-r+y_offset, x+r, y+r+y_offset)

    def next_move(self):
        if self.move_index >= len(PRESET):
            move = random.randint(1, 6)
        else:
            move = PRESET[self.move_index]

        if self.current_player == 1:
            start = self.p1_pos
            raw_target = min(start + move, 100)

            self.show_move_arrow(start, raw_target, "yellow")

            self.p1_pos = self.apply_chute_ladder(raw_target)

            self.current_player = 2
        else:
            start = self.p2_pos
            raw_target = min(start + move, 100)

            self.show_move_arrow(start, raw_target, "purple")

            self.p2_pos = self.apply_chute_ladder(raw_target)

            self.current_player = 1

        self.update_player_tokens()
        self.update_probabilities()

        self.move_index += 1

    def show_move_arrow(self, start_sq, end_sq, color):
        if self.move_arrow is not None:
            self.canvas.delete(self.move_arrow)
            self.move_arrow = None

        x1, y1 = self.square_center(start_sq)
        x2, y2 = self.square_center(end_sq)

        self.move_arrow = self.canvas.create_line(
            x1, y1, x2, y2,
            arrow=tk.LAST,
            width=3,
            fill=color
        )

    def apply_chute_ladder(self, pos):
        if pos in LADDERS:
            return LADDERS[pos]
        if pos in CHUTES:
            return CHUTES[pos]
        return pos

    def update_probabilities(self):
        p1_win, p2_win = compute_win_probabilities(
            self.p1_pos,
            self.p2_pos,
            self.current_player
        )

        self.prob_top.config(text=f"Player 1 (yellow) win prob: {p1_win * 100:.1f}%")
        self.prob_bottom.config(text=f"Player 2 (purple) win prob: {p2_win * 100:.1f}%")

root = tk.Tk()
app = ChutesLaddersDemo(root)
root.mainloop()
