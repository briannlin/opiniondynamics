import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time

class Person:
    """
    An abstraction of a person who holds an opinion.
    """
    def __init__(self, opinion: float):
        self.opinion = opinion

    def get_opinion(self) -> float:
        return self.opinion

    def set_opinion(self, opinion: float):
        self.opinion = opinion

    def __repr__(self):
        return f'Person({self.opinion})'


class VoterModel:
    """
    Representation of a voter model that, on every timeskip, probabilistically updates the opinion of
    every person in the model based on a probability calculated from the opinions of its nearest neighbors.
    """
    def __init__(self, n: int, m: int):
        self.n = n      # rows
        self.m = m      # cols
        self.matrix = [[Person(False) for _ in range(m)] for _ in range(n)]
        self.im = None

    def set_opinion(self, coords: (int, int), opinion: float):
        """
        Given a coordinate and opinion, sets the person's opinion who is at that coordinate to the
        given opinion.
        CHANGE - OPINIONS: PROBABILISTIC, NOT JUST BOOLEAN T/F
        """
        i, j = coords[0], coords[1]
        self.matrix[i][j].set_opinion(opinion)

    def _nearest_neighbors(self, coords: (int, int)) -> [(int, int)]:
        """
        Given a coordinate, returns a list of coordinates representing its nearest neighbors.
        CHANGE - HOW MANY NEIGHBORS TO CONSIDER
        """
        nearest_neighbors = []
        i, j = coords[0], coords[1]
        for _i in range(i - 1, i + 2):
            for _j in range(j - 1, j + 2):
                if _i < 0 or _j < 0 or _i >= self.n or _j >= self.m or (_i == i and _j == j):
                    continue
                else:
                    nearest_neighbors.append(self.matrix[_i][_j])
        return nearest_neighbors

    def update_opinion(self, coords: (int, int)):
        """
        Given a coordinate, updates the opinion of the person at that coordinate probabilistically
        based on the opinions of its nearest neighbors.
        CHANGE: LOGIC - HOW PEOPLE GET AFFECTED
        """
        i, j = coords
        opn = 0

        __i = 0
        count = 3
        range = 2
        start_i = random.randint(0, self.m)
        start_j = random.randint(0, self.n)
        while __i != count:
            _i = random.randint(start_i - range, start_i + range)
            _j = random.randint(start_j - range, start_j + range)
            if _i < self.m and _j < self.n:
                opn += self.matrix[_i][_j].get_opinion()
                __i += 1


        self.matrix[i][j].set_opinion(opn/count)

    def _display_opinion_matrix(self):
        """
        Displays the voter model. Red cells indicate False opinions, Blue cells indicate True opinions.
        """
        opinion_matrix = np.array([[self.matrix[i][j].get_opinion() for j in range(self.m)] for i in range(self.n)])

        if self.im == None:
            cim = plt.imread("rb_gradient.png")
            cim = cim[cim.shape[0] // 2, 8:740, :]
            cmap = mcolors.ListedColormap(cim)
            self.im = plt.imshow(opinion_matrix, cmap=cmap)

        self.im.set_data(opinion_matrix)
        plt.pause(0.05)


    def _is_over(self):
        """
        Returns whether the voter model is in a finished state or not.
        A state is considered finished when the model is taken over by a single opinion.
        """
        total_cells = self.n * self.m
        total_trues = 0
        for row in self.matrix:
            for person in row:
                total_trues += person.get_opinion()

        if total_trues == 0 or total_trues == total_cells:
            return True
        else:
            return False

    def _oneskip(self):
        """
        Perform one timeskip in the model, which randomly samples all the cells in the model and
        applies the probabilistic nearest-neighbour updating on each cell.
        """
        coords = []
        for i in range(self.n):
            for j in range(self.m):
                coords.append((i, j))

        random.shuffle(coords)
        for coord in coords:
            self.update_opinion(coord)

    def timeskip(self, t: int):
        """
        Given an amount of time t, performs t timeskips on the model and displays the live changes in state.
        """
        plt.title("Time = 0")
        self._display_opinion_matrix()
        for i in range(t):
            self._oneskip()
            plt.title(f"Time = {i+1}")
            self._display_opinion_matrix()

            if self._is_over():
                time.sleep(3)


if __name__ == '__main__':
    n = 100
    m = 100
    model = VoterModel(n, m)

    #mu, sigma = 0.5, 0.1
    #s = np.random.normal(mu, sigma, size=(n, m))
    for i in range(n):
        for j in range(m):
            #opinion = s[i][j]
            opinion = random.randint(0, 1)
            model.set_opinion((i, j), opinion)

    for row in model.matrix:
        print(row)


    #model.set_opinion((0, 0), 0)
    #model.set_opinion((n-1, m-1), 1)
    # model.set_opinion((3, 4), True)
    # model.set_opinion((2, 3), True)
    # model.set_opinion((3, 3), True)
    # model.set_opinion((4, 3), True)
    # model.set_opinion((8, 3), True)

    model.timeskip(1000)
    plt.show()