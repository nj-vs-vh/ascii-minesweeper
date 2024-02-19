import numpy as np
import math
from itertools import combinations
from string import ascii_uppercase
from numba import njit  # type: ignore
from tqdm import tqdm
from enum import Enum

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap, ScalarMappable

from typing import Callable, Generator, Tuple, Union, Optional, List


import ansi


ColIndex = Union[int, str]
RowIndex = int


class AsciiColumnRendering(Enum):
    ONE_CHAR = 1
    THREE_CHARS = 3

    def __call__(self, ch: str, fillable: bool = False) -> str:
        if self is self.ONE_CHAR:
            return ch
        if self is self.THREE_CHARS:
            if fillable:
                return ch * 3
            else:
                return " " + ch + " "
        else:
            raise ValueError(f"Unexpected self: {self}")


COL_RENDERING = AsciiColumnRendering.ONE_CHAR


class Board:

    def __init__(self, rows: int, cols: int, mines: int):
        self.rows = rows
        self.cols = cols
        self.mines = mines
        self.column_index_letters = ascii_uppercase[: self.cols]
        self.is_open = self._flat2plane(np.zeros(self.cells_count, dtype=bool))
        self.is_marked = self._flat2plane(np.zeros(self.cells_count, dtype=bool))
        self.is_pointer = self._flat2plane(np.zeros(self.cells_count, dtype=bool))

        self.board_created = False
        self.game_over = False

        self.mine_freq: Optional[np.ndarray] = None
        self.mine_freq_mask: Optional[np.ndarray] = None
        self.mine_total_combinations: Optional[int] = None

    def _init_board(self):
        is_mine_flat = np.zeros(self.cells_count, dtype=bool)
        is_mine_flat[np.random.choice(self.cells_count, self.mines, replace=False)] = (
            True
        )
        self.is_mine = self._flat2plane(is_mine_flat)

        self.near_mines = np.zeros((self.rows, self.cols), dtype=int)
        for i, j in self._board_idx():
            self.near_mines[i, j] = self.count_neighbors(i, j, values=self.is_mine)
        self.board_created = True

    def open_cell(self, i: RowIndex, j: ColIndex):
        j = self._validate_index(j, column=True)
        i = self._validate_index(i, column=False)

        if not self.board_created:
            while True:
                self._init_board()
                if not self.is_mine[i, j] and self.near_mines[i, j] == 0:
                    break

        self.is_open[i, j] = True
        if self.is_mine[i, j]:
            self.game_over = True
            for i, j in self._board_idx():
                if self.is_mine[i, j]:
                    self.is_open[i, j] = True
        elif self.near_mines[i, j] == 0:
            for i_n, j_n in self._neighborhood_idx(i, j):
                if not self.is_open[i_n, j_n]:
                    self.open_cell(i_n, j_n)

    BOARD_NOT_CREATED_MSG = ansi.modify(
        "\nOpen at least one cell first!\n", [ansi.TEXT.BOLD]
    )

    def mark_cell(self, i: RowIndex, j: ColIndex):
        j = self._validate_index(j, column=True)
        i = self._validate_index(i, column=False)
        if not self.board_created:
            print(self.BOARD_NOT_CREATED_MSG)
            return
        if not self.is_open[i, j]:
            self.is_marked[i, j] = True

    def unmark_cell(self, i: RowIndex, j: ColIndex):
        j = self._validate_index(j, column=True)
        i = self._validate_index(i, column=False)
        if not self.board_created:
            print(self.BOARD_NOT_CREATED_MSG)
            return
        if not self.is_open[i, j]:
            self.is_marked[i, j] = False

    def open_neighbors(self, i: RowIndex, j: ColIndex):
        j = self._validate_index(j, column=True)
        i = self._validate_index(i, column=False)
        if not self.board_created:
            print(self.BOARD_NOT_CREATED_MSG)
            return
        if not self.is_open[i, j]:
            return
        marked_neighbors = self.count_neighbors(i, j, values=self.is_marked)
        if marked_neighbors == self.near_mines[i, j]:
            for i_n, j_n in self._neighborhood_idx(i, j):
                if not self.is_marked[i_n, j_n]:
                    self.open_cell(i_n, j_n)

    def fast_forward(self):
        if not self.board_created:
            print(self.BOARD_NOT_CREATED_MSG)
            return
        if self.mine_freq is not None and self.fast_forward_safe:
            for i, j in self._board_idx():
                if self.mine_freq_mask[i, j]:
                    if self.mine_freq[i, j] == 0:
                        self.open_cell(i, j)
                    elif self.mine_freq[i, j] == self.mine_total_combinations:
                        self.mark_cell(i, j)
        while True:
            marked_before = np.sum(self.is_marked)
            open_before = np.sum(self.is_open)
            for i, j in self._board_idx():
                if not self.is_open[i, j] or self.near_mines[i, j] == 0:
                    continue
                near_unopen = self.count_neighbors(i, j, np.logical_not(self.is_open))
                if near_unopen == self.near_mines[i, j]:
                    for i_n, j_n in self._neighborhood_idx(i, j):
                        if not self.is_open[i_n, j_n]:
                            self.mark_cell(i_n, j_n)
            for i, j in self._board_idx():
                self.open_neighbors(i, j)
            if (
                np.sum(self.is_marked) == marked_before
                and np.sum(self.is_open) == open_before
            ):
                break

    @property
    def cells_count(self) -> int:
        return self.rows * self.cols

    @property
    def mines_left(self) -> int:
        return self.mines - np.sum(self.is_marked)

    @property
    def cells_left(self) -> int:
        return self.cells_count - (np.sum(self.is_marked) + np.sum(self.is_open))

    def info(self):
        if not self.board_created:
            print(self.BOARD_NOT_CREATED_MSG)
            return
        print(
            "\n"
            + ansi.modify(
                f"Mines left: {self.mines_left} / {self.mines}\n"
                + f"Cells left: {self.cells_left} / {self.rows * self.cols}",
                codes=[ansi.TEXT.BOLD],
            )
            + "\n"
        )

    ASCII_GREYSCALE = " ·+|┼╬▒▓█"

    def mine_prob_colormap(self, prob: float):
        float_idx = len(self.ASCII_GREYSCALE) * prob - 1e-6
        int_idx = int(float_idx)
        codes = [ansi.TEXT.YELLOW]
        if float_idx - int_idx > 0.5:
            codes += [ansi.TEXT.BOLD]
        return ansi.modify(COL_RENDERING(self.ASCII_GREYSCALE[int_idx]), codes)

    def evaluate_probabilities(self) -> None:
        if not self.board_created:
            print(self.BOARD_NOT_CREATED_MSG)
            return

        MAX_BRUTEFORCE_CHOICES = int(5e6)

        total_choices = math.comb(self.cells_left, self.mines_left)

        if total_choices < MAX_BRUTEFORCE_CHOICES:
            self.fast_forward_safe = True
            n_trial = total_choices
            trial_indices_generator = combinations
        else:
            print(
                ansi.modify(
                    f"The search space is too large, capping at {MAX_BRUTEFORCE_CHOICES}!",
                    [ansi.BACK.RED],
                )
            )
            self.fast_forward_safe = False
            n_trial = int(min(total_choices / 2, MAX_BRUTEFORCE_CHOICES))
            rng = np.random.default_rng()

            batch_size = 1000
            n_batch = n_trial // batch_size

            n_trial = n_batch * batch_size

            def trial_indices_generator(indices, number):  # type: ignore
                indices_size = indices.size
                # for _ in range(n_trial):
                #     yield indices[rng.integers(0, indices_size, size=number)]
                for _ in range(n_batch):
                    batch = rng.integers(0, indices_size, size=(batch_size, number))
                    for rand_choice in batch:
                        yield indices[rand_choice]

        is_trial = np.logical_not(self.is_open + self.is_marked)
        is_affected = np.zeros_like(is_trial, dtype=bool)
        for i, j in self._board_idx():
            is_affected[i, j] = self.count_neighbors(i, j, is_trial) > 0
        is_affected = is_affected * self.is_open

        # all currently marked mines are treated as true
        # all hypothetical mines will be added to this array in-place
        is_mine_trial = self.is_marked.flatten()

        neighbor_counter = self.get_fast_neighbor_counter()
        rows = self.rows
        cols = self.cols
        near_mines = self.near_mines

        @njit
        def is_mine_trial_valid(is_mine_trial: np.ndarray) -> bool:
            for i in range(rows):
                for j in range(cols):
                    if is_affected[i, j]:
                        if neighbor_counter(i, j, is_mine_trial) != near_mines[i, j]:
                            return False
            return True

        trial_idx_flat = np.arange(self.cells_count)[is_trial.flatten()]
        trial_freq_flat = np.zeros_like(trial_idx_flat, dtype=int)
        total_valid_combinations = 0
        ii = np.arange(trial_idx_flat.size)
        for mines_ii in tqdm(
            trial_indices_generator(ii, self.mines_left), total=n_trial
        ):
            mines_ii = np.array(mines_ii)  # type: ignore
            mines_idx = trial_idx_flat[mines_ii]

            is_mine_trial[mines_idx] = True
            is_mine_trial.resize((self.rows, self.cols))
            if is_mine_trial_valid(is_mine_trial):
                trial_freq_flat[mines_ii] += 1
                total_valid_combinations += 1
            is_mine_trial.resize((self.cells_count))
            is_mine_trial[mines_idx] = False

        freq = np.zeros((self.cells_count), dtype=int)
        freq[trial_idx_flat] = trial_freq_flat
        freq_mask = np.zeros((self.cells_count), dtype=bool)
        freq_mask[trial_idx_flat] = True
        self.mine_freq_mask = freq_mask.reshape((self.rows, self.cols))
        self.mine_freq = freq.reshape((self.rows, self.cols))
        if total_valid_combinations == 0:
            self.mine_total_combinations = 1
            print("No valid combinations found!")
        else:
            self.mine_total_combinations = total_valid_combinations

    def set_pointers(self, pointers: List[Tuple[int, str]]):
        self.is_pointer[:] = False
        for i, j in pointers:
            i = self._validate_index(i, column=False)
            j_numeric = self._validate_index(j, column=True)
            self.is_pointer[i, j_numeric] = True

    @property
    def game_winned(self) -> bool:
        if not self.board_created:
            return False
        marked_total = np.sum(self.is_marked)
        open_total = np.sum(self.is_open)
        return (
            marked_total == self.mines
            and open_total + marked_total == self.rows * self.cols
        )

    def _validate_index(
        self, index: Union[RowIndex, ColIndex], column: bool = True
    ) -> int:
        if isinstance(index, str):
            index = index.upper()
            if index == "":
                raise ValueError(f"Empty string passed as index")
            index = self.column_index_letters.index(index)
        if index < 0 or index >= (self.cols if column else self.rows):
            raise ValueError(f"Index out of bounds")
        return index

    def _board_idx(self) -> Generator[Tuple[int, int], None, None]:
        for i in range(self.rows):
            for j in range(self.cols):
                yield i, j

    def _neighborhood_idx(
        self, i: int, j: int
    ) -> Generator[Tuple[int, int], None, None]:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue  # do not count center
                if 0 <= i + di < self.rows and 0 <= j + dj < self.cols:
                    yield i + di, j + dj

    def count_neighbors(self, i: int, j: int, values: np.ndarray) -> int:
        return sum(int(values[i_n, j_n]) for i_n, j_n in self._neighborhood_idx(i, j))

    def get_fast_neighbor_counter(self) -> Callable[[int, int, np.ndarray], int]:
        rows: int = self.rows
        cols: int = self.cols

        @njit
        def fast_neighbor_counter(i: int, j: int, values: np.ndarray) -> int:
            summ = 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue  # do not count center
                    if 0 <= i + di < rows and 0 <= j + dj < cols:
                        summ += values[i + di, j + dj]
            return summ

        return fast_neighbor_counter

    def _flat2plane(self, flattened_plane: np.ndarray) -> np.ndarray:
        return flattened_plane.reshape((self.rows, self.cols))

    def _render_field(self, cell_char: Callable[[int, int], str]):
        max_number_len = len(str(self.rows - 1))
        letters_legend = (
            " " * max_number_len
            + " "
            + "".join(COL_RENDERING(s) for s in self.column_index_letters)
        )
        print(letters_legend)

        ruler_ansi_codes = [ansi.TEXT.BLUE]
        top_ruler = " " * max_number_len + ansi.modify(
            "╔" + COL_RENDERING("═", fillable=True) * self.cols + "╗",
            codes=ruler_ansi_codes,
        )
        bot_ruler = " " * max_number_len + ansi.modify(
            "╚" + COL_RENDERING("═", fillable=True) * self.cols + "╝",
            codes=ruler_ansi_codes,
        )
        vert_ruler = ansi.modify("║", codes=ruler_ansi_codes)

        print(top_ruler)
        for i in range(self.rows):
            print(str(i).rjust(max_number_len) + vert_ruler, end="")
            for j in range(self.cols):
                char = cell_char(i, j)
                print(char, end="")
            print(vert_ruler + str(i))
        print(bot_ruler)
        print(letters_legend)

    def render_mines(self):
        self._render_field(lambda i, j: "m" if self.is_mine[i, j] else " ")

    NEAR_MINES_COL = {
        1: ansi.TEXT.BLUE,
        2: ansi.TEXT.GREEN,
        3: ansi.TEXT.RED,
        4: ansi.TEXT.MAGENTA,
        5: ansi.TEXT.YELLOW,
        6: ansi.TEXT.YELLOW,
        7: ansi.TEXT.YELLOW,
        8: ansi.TEXT.YELLOW,
    }

    def render(self):

        def cell_char(i, j) -> str:
            if self.board_created:
                if self.is_open[i, j]:
                    if self.is_mine[i, j]:
                        ch = ansi.modify(COL_RENDERING("m"), [ansi.BACK.RED])
                    else:
                        near_mines = self.near_mines[i, j]
                        ch = (
                            COL_RENDERING("░", fillable=True)
                            if near_mines == 0
                            else ansi.modify(
                                COL_RENDERING(str(near_mines)),
                                codes=[self.NEAR_MINES_COL[near_mines]],
                            )
                        )
                elif self.is_marked[i, j]:
                    ch = COL_RENDERING("×")
                else:
                    if self.mine_freq is None:
                        ch = COL_RENDERING(" ", fillable=True)
                    else:
                        scaled_mine_prob = (
                            self.mine_freq[i, j] / self.mine_total_combinations
                        )
                        ch = self.mine_prob_colormap(scaled_mine_prob)
            else:
                ch = COL_RENDERING(" ", fillable=True)

            if self.is_pointer[i, j]:
                ch = ansi.modify(ch, [ansi.BACK.MAGENTA, ansi.TEXT.BOLD])
            return ch

        self._render_field(cell_char)

        if self.mine_freq is not None:
            freq_values = np.unique(self.mine_freq)
            mine_prob_values = freq_values / self.mine_total_combinations
            # scaled_prob_values = freq_values / np.max(self.mine_freq)

            print(ansi.modify("Mine probability:\n", codes=[ansi.TEXT.BOLD]))
            print(
                "".join(
                    [
                        s * 6
                        for s in [
                            self.mine_prob_colormap(sp) for sp in mine_prob_values
                        ]
                    ]
                )
            )
            print("".join([f" {p:.2f} " for p in mine_prob_values]))

    def render_as_plot(self):
        max_edge_size_px = 500
        px_per_rowcol = max_edge_size_px / max(self.cols, self.rows)
        dpi = 96
        additional_cols = 2 if self.mine_freq is not None else 0
        fig, ax = plt.subplots(
            figsize=(
                px_per_rowcol * (self.cols + additional_cols) / dpi,
                px_per_rowcol * self.rows / dpi,
            ),
            dpi=dpi,
        )

        GRID_COLOR = "#d4d4d4"
        OPEN_CELL = "#D5D9E0"
        NEAR_MINES = {
            1: "#4AA5F0",
            2: "#8CC265",
            3: "#E05561",
            4: "#C162DE",
            5: "#D18F52",
            6: "#D18F52",
            7: "#D18F52",
            8: "#D18F52",
        }

        # grid and labels
        for i, j in self._board_idx():
            ax.axhline(i - 0.5, color=GRID_COLOR)
            ax.axvline(j - 0.5, color=GRID_COLOR)

        ax.tick_params(length=0.0)
        ax.set_xlim(left=-0.5, right=self.cols - 0.5)
        ax.set_xticks(np.arange(self.cols))
        ax.set_xticklabels(self.column_index_letters)

        ax.set_ylim(bottom=-0.5, top=self.rows - 0.5)
        ax.set_yticks(np.arange(self.rows))
        ax.set_yticklabels([str(t) for t in range(self.rows)[::-1]])

        def board2screen(i: int, j: int):
            """Returns center of the cell in Axis coordinates"""
            return j, self.rows - i - 1

        # cell coloring
        rects = []
        for i, j in self._board_idx():
            if self.is_open[i, j]:
                si, sj = board2screen(i, j)
                rects.append(Rectangle((si - 0.5, sj - 0.5), 1, 1))
        ax.add_collection(PatchCollection(rects, facecolor=OPEN_CELL, edgecolor=None))

        # numbers in cells and marked mines
        # determining text position in data units
        test_font_size = 10
        test_text = ax.text(0.5, 0.5, "test", {"size": test_font_size})
        r = fig.canvas.get_renderer()

        data_zero_disp_x, data_zero_disp_y = ax.transData.transform((0, 0))

        def get_text_width_height(t: plt.Text) -> Tuple[float, float]:
            bb = t.get_window_extent(renderer=r)
            return ax.transData.inverted().transform(
                (data_zero_disp_x + bb.width, data_zero_disp_y + bb.height)
            )

        target_text_height = 0.7
        _, test_text_height = get_text_width_height(test_text)
        font_size = test_font_size * target_text_height / np.abs(test_text_height)
        test_text.set_visible(False)

        for i, j in self._board_idx():
            if self.is_open[i, j] and self.near_mines[i, j] > 0:
                si, sj = board2screen(i, j)
                ax.text(
                    si,
                    sj,
                    str(self.near_mines[i, j]),
                    {"size": font_size},
                    color=NEAR_MINES[self.near_mines[i, j]],
                    ha="center",
                    va="center",
                )
            if self.is_marked[i, j]:
                si, sj = board2screen(i, j)
                pad = 0.2
                flag_center_y = 0.65
                ax.add_line(
                    Line2D(
                        xdata=np.array([pad, 1 - pad, 0.5, 0.5]) + si - 0.5,
                        ydata=np.array([pad, pad, pad, flag_center_y]) + sj - 0.5,
                        color="black",
                    ),
                )
                ax.add_patch(
                    Circle(
                        (si, sj - 0.5 + flag_center_y),
                        radius=(1 - flag_center_y - pad),
                        facecolor="red",
                        zorder=1000,
                    )
                )

        if self.mine_freq is not None:
            cmap = get_cmap("hot").reversed()
            # for i, j in self._board_idx():
            rects = []
            for i, j in self._board_idx():
                if self.mine_freq_mask[i, j]:
                    si, sj = board2screen(i, j)
                    ax.add_patch(
                        Rectangle(
                            (si - 0.5, sj - 0.5),
                            1,
                            1,
                            facecolor=cmap(
                                (self.mine_freq[i, j] / self.mine_total_combinations)
                                - 1e-6
                            ),
                        )
                    )

            cbar = fig.colorbar(ScalarMappable(cmap=cmap))
            cbar.set_label("Mine probability")

        plt.show()
