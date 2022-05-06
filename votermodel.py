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
        self.affected_by = []

    def get_opinion(self) -> float:
        return self.opinion

    def set_opinion(self, opinion: float):
        self.opinion = opinion

    def get_affected_by(self):
        return self.affected_by

    def add_affected_by(self, coords: (int, int)):
        self.affected_by.append(coords)

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
        CHANGE - HOW MANY NEIGHBORS TO CONSIDER (SCOPE OF INFLUENCE)
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
        CHANGE: Travel: stochastically sample two people
        CHANGE: scope of influence
        """
        i, j = coords
        affecting_neighbors = self.matrix[i][j].get_affected_by()
        num_neighbors = len(affecting_neighbors)
        num_true = sum(self.matrix[affector[0]][affector[1]].get_opinion() for affector in affecting_neighbors)
        Pconvert = 0.1*((num_true / num_neighbors) - 0.5)
        opinion = self.matrix[i][j].get_opinion()
        if (opinion > 1 or opinion < 0) and (abs(opinion+Pconvert) - abs(opinion)) > 0:
            Pconvert = 0
        self.matrix[i][j].set_opinion(opinion + Pconvert)

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
        plt.pause(0.1)

        # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "blue"])
        # plt.imshow(opinion_matrix, interpolation='none', cmap=cmap)
        #
        # colors = [(.8, .8, 1), (1, .8, .8)]  # first color is black, last is red
        # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("Custom", colors, N=20)
        # norm = plt.Normalize(opinion_matrix.min(), opinion_matrix.max())
        # rgba = cmap(norm(opinion_matrix))
        #
        # for i in range(self.n):
        #     for j in range(self.m):
        #         if opinion_matrix[i][j] < 0.5:
        #             # Blue
        #             blue = .25 + opinion_matrix[i][j]
        #             red = 0
        #             green = 0
        #         else:
        #             # Red
        #             red = 1.5 - opinion_matrix[i][j]
        #             green = 0
        #             blue = 0
        #
        #         rgba[i, j, :3] = red, green, blue
        #
        # plt.imshow(rgba, cmap=cmap)
        #plt.show()
        #plt.pause(0.01)


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
    n = 200
    m = 200
    model = VoterModel(n, m)

    # Initialize opinions to be random
    mu, sigma = 0.5, 0.1
    s = np.random.normal(mu, sigma, size=(n, m))
    for i in range(n):
        for j in range(m):
            opinion = s[i][j]
            model.set_opinion((i, j), opinion)

    # Loop through all cells, getting random # btwn 0 & 1 to determine range of influence for each cell
    for i in range(n):
        for j in range(m):
            pr = random.random()
            if 0 <= pr <= 0.0025:
                roi = ((((((((min(n, m) // 10))))))))
            elif 0.02 < pr <= 0.6:
                roi = 1
            elif 0.6 < pr <= 0.85:
                roi = 2
            else:
                roi = 3

            # Cell (k, l), a cell in ROI of cell (i, j), is affected by cell (i, j)
            for k in range(i - roi, i + roi + 1):
                for l in range(j - roi, j + roi + 1):
                    if k >= 0 and l >= 0 and k < n and l < m and (k != i or l != j):
                        person = model.matrix[k][l]
                        person.add_affected_by((i, j))


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
