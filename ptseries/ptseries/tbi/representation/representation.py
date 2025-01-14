import matplotlib.pyplot as plt
import numpy as np


class Drawer:
    def __init__(self):
        self.mode_i_depth = None  # current depth of each mode
        self.j = 0
        self.spacing = 2  # offset between consecutive loops

        self.color = "black"  # color of the representation lines

    def BS(self, i0, i1, idx, add_detail=True):
        """Draw a beam splitter of size 1 x 1 that connects modes i0 and i1 (i1 >i0)"""

        spacing = 0
        if i0 == 0:
            spacing = self.spacing

        j = max(self.mode_i_depth) + spacing
        larms = 0.5
        beta = 12
        x = np.linspace(-larms + j + 0.5, larms + j + 0.5, 50)
        y2 = -i0 + 1 - (i1 - i0) / (1 + np.exp(-beta * (x - j - 0.5))) - 1
        y1 = y2[::-1]
        plt.plot(x, y1, self.color)
        plt.plot(x, y2, self.color)

        if add_detail:
            if (i1 - i0) % 2 == 1:
                plt.hlines(-i0 - (i1 - i0) / 2, -larms + j + 0.5, larms + j + 0.5, color=self.color)
            plt.text(
                j + 1 + 0.05 * self.input_length,
                -i0 - 0.3,
                r"$\theta_{}$".format("{" + str(idx + 1) + "}"),
                c=self.color,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=int(20 - self.input_length),
            )

        self.j += 1

        mode_i_depth_prev = self.mode_i_depth.copy()

        for k in range(i0, i1 + 1):
            self.mode_i_depth[k] = j + 1

        if i0 == 0:
            plt.hlines(0, mode_i_depth_prev[0], j, color=self.color)
            plt.hlines(-i1, mode_i_depth_prev[i1], j, color=self.color)

        for i, depth_i in enumerate(self.mode_i_depth):
            depth_max = max(self.mode_i_depth)
            if i1 == i0 + 1:
                if depth_i > 0 and i != i0 and i != i1:
                    plt.hlines(-i, mode_i_depth_prev[i], depth_max, color=self.color)

            else:
                if depth_i > 0 and i != i0 and i != i1:  # connection of far neigh
                    plt.hlines(-i, mode_i_depth_prev[i], depth_max - 1, color=self.color)
                    if i > i0 and i < i1:
                        plt.hlines(-i, depth_max - 1, depth_max, linestyles="dashed", color=self.color, alpha=0.3)
                    else:
                        plt.hlines(-i, depth_max - 1, depth_max, color=self.color)

            if mode_i_depth_prev[i] == 0 and depth_i > 0:
                plt.hlines(-i, 0, depth_i - 1, color=self.color)

    def end(self, padding=0):
        """Fill the modes at the end of the circuit so that they all have the same length"""
        depth_max = max(self.mode_i_depth) + padding
        for i, depth_i in enumerate(self.mode_i_depth):
            if depth_i > 0 and depth_i < depth_max:
                plt.hlines(-i, depth_i, depth_max, color=self.color)

    def start(self, padding=0):
        """Fill the modes at the begining of the circuit so that they all have the same length"""
        for i, depth_i in enumerate(self.mode_i_depth):
            if depth_i > 0:
                plt.hlines(-i, -padding, 0, color=self.color)

    def add_inputs(self, input_state):
        """Print the input photons at the beginning of the circuit"""

        for i in range(len(input_state)):
            mode = input_state[i]
            plt.text(
                -1.5,
                -i,
                "|" + str(mode) + ">",
                c=self.color,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=int(20 - self.input_length),
            )

    def draw(self, structure, input_state, padding=0):
        """Draw the circuit given an input state and a structure, up to 30 modes

        If the number of modes is greater or equal to 10, details are ommitted
        """

        self.input_length = len(input_state)
        if self.input_length > 30:
            raise Exception("Cannot draw an interferometer with more than 30 modes.")

        # Don't add detail if the number of modes is too large
        if self.input_length < 10:
            add_detail = True
        else:
            add_detail = False

        for idx, modes in enumerate(structure):
            self.BS(modes[0], modes[1], idx, add_detail=add_detail)

        if add_detail:
            self.add_inputs(input_state)

        self.start(padding=padding)
        self.end(padding=padding)

        plt.axis("off")

    def get_structure(self, n_modes, n_loops, loop_lengths):
        """
        Compute the structure of the circuit,i.e, a list of ordered tuples where each tuple (i,j) represents
        the interference between modes i and j.
        """
        self.mode_i_depth = [0] * n_modes
        structure = []
        for loop in range(n_loops):
            delay = loop_lengths[loop]
            for i in range(n_modes - delay):
                structure.append([i, i + delay])

        return structure
