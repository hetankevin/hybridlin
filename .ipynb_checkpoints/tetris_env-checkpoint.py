import numpy as np

WIDTH = 6

## This code was obtained from a graduate-level reinforcement learning class.
## As I understand it, the original code may have been written in C++ here:
##     https://melax.github.io/tetris/tetris.html
## The code was adapted to Python as part of the class materials.
## Divulging more information may violate the conference's double-blind reviewing process.

class TetrisEnv:
    def __init__(self, seed=None):
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)
        self.game_area = 0
        self.piece = (self.rng.integers(low=0, high=4) << WIDTH) + self.rng.integers(low=0, high=3) + 1
        self.height = 0

    # Transform the piece representation from 2*WIDTH bits to 4 bits
    @staticmethod
    def piece_transform(piece):
        low = piece & 3
        high = (piece & (3 << WIDTH)) >> (WIDTH - 2)
        return low + high

    def reset(self):
        self.game_area = 0
        self.piece = (self.rng.integers(low=0, high=4) << WIDTH) + self.rng.integers(low=0, high=3) + 1
        self.height = 0
        return self.game_area, self.piece_transform(self.piece)

    def get_state(self):
        return self.game_area, self.piece_transform(self.piece)

    def get_height(self):
        return self.height

    @staticmethod
    def rotate(p, rotation):
        # rotate the piece clockwise
        while rotation > 0:
            q = p >> WIDTH
            p = (2 if p & 1 != 0 else 0) + (2 << WIDTH if p & 2 != 0 else 0) + (1 << WIDTH if q & 2 != 0 else 0) + (
                1 if q & 1 != 0 else 0)
            rotation = rotation - 1
        if p % (1 << WIDTH) == 0:
            p >>= WIDTH
        return p

    def step(self, position, rotation):
        assert (0 <= round(position) <= WIDTH - 2) and (
                    0 <= round(rotation) <= 3), "Error: The action input is invalid!"
        position = round(position)
        rotation = round(rotation)
        piece = self.rotate(self.piece, rotation)  # rotate the piece
        piece = piece << position  # move to the position
        # drop down the piece
        while (piece & self.game_area) or ((piece << WIDTH) & self.game_area):
            piece = piece << WIDTH
        t = piece | self.game_area

        # if the top row is full, then remove the row
        if (t & (((1 << WIDTH) - 1) << WIDTH)) == (((1 << WIDTH) - 1) << WIDTH):
            t = (t & ((1 << WIDTH) - 1)) | ((t >> (2 * WIDTH)) << WIDTH)

        # if the bottom row is full, then remove the row
        if (t & ((1 << WIDTH) - 1)) == ((1 << WIDTH) - 1):
            t >>= WIDTH

        self.game_area = t
        self.piece = (self.rng.integers(low=0, high=4) << WIDTH) + self.rng.integers(low=0, high=3) + 1

        loss = 0
        while (self.game_area >> (2 * WIDTH)) != 0:
            self.game_area = self.game_area >> WIDTH
            loss = loss + 1
        assert (self.game_area < (1 << 2 * WIDTH)), "Unknown Error!"

        self.height = self.height + loss
        reward = -loss

        return self.game_area, self.piece_transform(self.piece), reward
