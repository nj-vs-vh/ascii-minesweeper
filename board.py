import numpy as np
import math
from itertools import combinations
from string import ascii_uppercase
from numba import njit
from tqdm import tqdm

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap, ScalarMappable

from typing import Callable, Generator, Tuple, Union, Optional


import ansi


ColIndex = Union[int, str]
RowIndex = int


class Board:

    def __init__(self, rows: int, cols: int, mines: int):
        self.rows = rows
        self.cols = cols
        self.mines = mines
        self.column_index_letters = ascii_uppercase[:self.cols]

        self.board_created = False
        self.game_over = False

        self.mine_freq = None
        self.mine_freq_mask = None
        self.mine_total_combinations = None

    def _init_board(self):
        is_mine_flat = np.zeros(self.cells_count, dtype=bool)
        is_mine_flat[np.random.choice(self.cells_count, self.mines, replace=False)] = True
        self.is_mine = self._flat2plane(is_mine_flat)
        self.is_marked = self._flat2plane(np.zeros(self.cells_count, dtype=bool))
        self.is_open = self._flat2plane(np.zeros(self.cells_count, dtype=bool))

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

    BOARD_NOT_CREATED_MSG = ansi.modify('\nOpen at least one cell first!\n', [ansi.TEXT.BOLD])

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

    def play_forward(self):
        if not self.board_created:
            print(self.BOARD_NOT_CREATED_MSG)
            return
        if self.mine_freq is not None:
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
            if np.sum(self.is_marked) == marked_before and np.sum(self.is_open) == open_before:
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
            '\n'
            + ansi.modify(
                f'Mines left: {self.mines_left} / {self.mines}\n'
                + f'Cells left: {self.cells_left} / {self.rows * self.cols}',
                codes=[ansi.TEXT.BOLD]
            ) + '\n'
        )

    ASCII_GREYSCALE = ' ·+|┼╬▒▓█'

    def mine_prob_colormap(self, prob: float):
        float_idx = len(self.ASCII_GREYSCALE) * prob - 1e-6
        int_idx = int(float_idx)
        codes = [ansi.TEXT.YELLOW]
        if float_idx - int_idx > 0.5:
            codes += [ansi.TEXT.BOLD]
        return ansi.modify(self.ASCII_GREYSCALE[int_idx], codes)

    def evaluate_probabilities(self):
        if not self.board_created:
            print(self.BOARD_NOT_CREATED_MSG)
            return
        
        total_choices = math.comb(self.cells_left, self.mines_left)
        if total_choices > 1e8:
            print(ansi.modify("\nToo much choices to brute-force :(\n", [ansi.TEXT.BOLD]))
            return

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
        for mines_ii in tqdm(combinations(ii, self.mines_left), total=total_choices):
            mines_ii = np.array(mines_ii)
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
            print(ansi.modify('No valid combinations found!', [ansi.BACK.RED]))
        else:
            self.mine_total_combinations = total_valid_combinations

    @property
    def game_winned(self) -> bool:
        if not self.board_created:
            return False
        marked_total = np.sum(self.is_marked)
        open_total = np.sum(self.is_open)
        return marked_total == self.mines and open_total + marked_total == self.rows * self.cols

    def _validate_index(self, index: Union[RowIndex, ColIndex], column: bool = True) -> int:
        if isinstance(index, str):
            index = index.upper()
            if index == '':
                raise ValueError(f"Empty string passed as index")
            index = self.column_index_letters.index(index)
        if index < 0 or index >= (self.cols if column else self.rows):
            raise ValueError(f"Index out of bounds")
        return index

    def _board_idx(self) -> Generator[Tuple[int, int], None, None]:
        for i in range(self.rows):
            for j in range(self.cols):
                yield i, j

    def _neighborhood_idx(self, i: int, j: int) -> Generator[Tuple[int, int], None, None]:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue  # do not count center
                if 0 <= i + di < self.rows and 0 <= j + dj < self.cols:
                    yield i + di, j + dj

    def count_neighbors(self, i: int, j: int, values: np.ndarray) -> int:
        return sum(
            int(values[i_n, j_n]) for i_n, j_n in self._neighborhood_idx(i, j)
        )

    def get_fast_neighbor_counter(self):
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
        print(' ' * max_number_len + ' ' + self.column_index_letters)

        ruler_ansi_codes = [ansi.TEXT.BLUE]
        top_ruler = ansi.modify('╔' + '═' * self.cols + '╗', codes=ruler_ansi_codes)
        bot_ruler = ansi.modify('╚' + '═' * self.cols + '╝', codes=ruler_ansi_codes)
        vert_ruler = ansi.modify('║', codes=ruler_ansi_codes)

        print(' ' * max_number_len + top_ruler)
        for i in range(self.rows):
            print(str(i).rjust(max_number_len) + vert_ruler, end='')
            for j in range(self.cols):
                char = cell_char(i, j)
                print(char, end='')
            print(vert_ruler + str(i))
        print(' ' * max_number_len + bot_ruler)
        print(' ' * max_number_len + ' ' + self.column_index_letters)

    def render_mines(self):
        self._render_field(lambda i, j: 'm' if self.is_mine[i, j] else ' ')

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

    def render(self, pointer: Optional[Tuple[int, int]] = None):
        if pointer is not None:
            pointer = (
                self._validate_index(pointer[0], column=False),
                self._validate_index(pointer[1], column=True),
            )

        def cell_char(i, j) -> str:
            if self.board_created:
                if self.is_open[i, j]:
                    if self.is_mine[i, j]:
                        ch = ansi.modify('m', [ansi.BACK.RED])
                    else:
                        near_mines = self.near_mines[i, j]
                        ch = (
                            '░'
                            if near_mines == 0
                            else ansi.modify(str(near_mines), codes=[self.NEAR_MINES_COL[near_mines]])
                        )
                elif self.is_marked[i, j]:
                    ch = '×'
                else:
                    if self.mine_freq is None:
                        ch = ' '
                    else:
                        scaled_mine_prob = self.mine_freq[i, j] / self.mine_total_combinations
                        ch = self.mine_prob_colormap(scaled_mine_prob)
            else:
                ch = ' '

            if pointer is None or i != pointer[0] or j != pointer[1]:
                return ch
            else:
                ch = '·' if ch == ' ' else ch
                return ansi.modify(ch, codes=[ansi.TEXT.RED, ansi.TEXT.BOLD])
            
        self._render_field(cell_char)

        if self.mine_freq is not None:
            freq_values = np.unique(self.mine_freq)
            mine_prob_values = freq_values / self.mine_total_combinations
            # scaled_prob_values = freq_values / np.max(self.mine_freq)

            print(ansi.modify('Mine probability:\n', codes=[ansi.TEXT.BOLD]))
            print(''.join([s * 6 for s in [self.mine_prob_colormap(sp) for sp in mine_prob_values]]))
            print(''.join([f' {p:.2f} ' for p in mine_prob_values]))

    def render_as_plot(self):
        max_edge_size_px = 500
        px_per_rowcol = max_edge_size_px / max(self.cols, self.rows)
        dpi = 96
        additional_cols = 2 if self.mine_freq is not None else 0
        fig, ax = plt.subplots(
            figsize=(
                px_per_rowcol * (self.cols + additional_cols) / dpi,
                px_per_rowcol * self.rows / dpi
            ),
            dpi=dpi,
        )

        GRID_COLOR = '#d4d4d4'
        OPEN_CELL = '#D5D9E0'
        NEAR_MINES = {
            1: '#4AA5F0',
            2: '#8CC265',
            3: '#E05561',
            4: '#C162DE',
            5: '#D18F52',
            6: '#D18F52',
            7: '#D18F52',
            8: '#D18F52',
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
        test_text = ax.text(0.5, 0.5, 'test', {'size': test_font_size})
        r = fig.canvas.get_renderer()

        data_zero_disp_x, data_zero_disp_y = ax.transData.transform((0, 0))

        def get_text_width_height(t: plt.Text) -> Tuple[float, float]:
            bb = t.get_window_extent(renderer=r)
            return ax.transData.inverted().transform(
                (data_zero_disp_x + bb.width, data_zero_disp_y+bb.height)
            )

        target_text_height = 0.7
        _, test_text_height = get_text_width_height(test_text)
        font_size = test_font_size * target_text_height / np.abs(test_text_height)
        test_text.set_visible(False)

        for i, j in self._board_idx():
            if self.is_open[i, j] and self.near_mines[i, j] > 0:
                si, sj = board2screen(i, j)
                ax.text(
                    si, sj, str(self.near_mines[i, j]),
                    {'size': font_size},
                    color=NEAR_MINES[self.near_mines[i, j]],
                    ha='center',
                    va='center',
                )
            if self.is_marked[i, j]:
                si, sj = board2screen(i, j)
                pad = 0.2
                flag_center_y = 0.65
                ax.add_line(
                    Line2D(
                        xdata=np.array([pad, 1 - pad, 0.5, 0.5]) + si - 0.5,
                        ydata=np.array([pad, pad, pad, flag_center_y]) + sj - 0.5,
                        color='black',
                    ),
                )
                ax.add_patch(
                    Circle(
                        (si, sj - 0.5 + flag_center_y), radius=(1 - flag_center_y - pad),
                        facecolor='red', zorder=1000,
                    )
                )
        
        if self.mine_freq is not None:
            cmap = get_cmap('hot').reversed()
            # for i, j in self._board_idx():
            rects = []
            for i, j in self._board_idx():
                if self.mine_freq_mask[i, j]:
                    si, sj = board2screen(i, j)
                    ax.add_patch(
                        Rectangle(
                            (si - 0.5, sj - 0.5), 1, 1,
                            facecolor=cmap((self.mine_freq[i, j] / self.mine_total_combinations) - 1e-6)
                        )
                    )

            cbar = fig.colorbar(ScalarMappable(cmap=cmap))
            cbar.set_label('Mine probability')

        plt.show()
