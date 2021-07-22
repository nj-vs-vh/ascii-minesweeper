import numpy as np
import math
from itertools import combinations
from string import ascii_uppercase
from numba import njit
from tqdm import tqdm

from typing import Callable, Generator, Tuple, Union, Optional, Iterable


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
        self.mine_total_combinations = None

    def _init_board(self):
        is_mine_flat = np.zeros(self.cells_count, dtype=bool)
        is_mine_flat[np.random.choice(self.cells_count, self.mines, replace=False)] = True
        self.is_mine = self._flat2plane(is_mine_flat)
        self.is_marked = self._flat2plane(np.zeros(self.cells_count, dtype=bool))
        self.is_open = self._flat2plane(np.zeros(self.cells_count, dtype=bool))

        self.near_mines = np.zeros((self.rows, self.cols), dtype=int)
        for i in range(self.rows):
            for j in range(self.cols):
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
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.is_mine[i, j]:
                        self.is_open[i, j] = True
        elif self.near_mines[i, j] == 0:
            for i_n, j_n in self._neighborhood(i, j):
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
            for i_n, j_n in self._neighborhood(i, j):
                if not self.is_marked[i_n, j_n]:
                    self.open_cell(i_n, j_n)

    def play_forward(self):
        if not self.board_created:
            print(self.BOARD_NOT_CREATED_MSG)
            return
        while True:
            marked_before = np.sum(self.is_marked)
            for i in range(self.rows):
                for j in range(self.cols):
                    if not self.is_open[i, j] or self.near_mines[i, j] == 0:
                        continue
                    near_unopen = self.count_neighbors(i, j, np.logical_not(self.is_open))
                    if near_unopen == self.near_mines[i, j]:
                        for i_n, j_n in self._neighborhood(i, j):
                            if not self.is_open[i_n, j_n]:
                                self.mark_cell(i_n, j_n)
            for i in range(self.rows):
                for j in range(self.cols):
                    self.open_neighbors(i, j)
            if np.sum(self.is_marked) - marked_before == 0:
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
            print(ansi.modify("\nToo much choices, probably better just guess :(\n", [ansi.TEXT.BOLD]))
            return

        is_trial = np.logical_not(self.is_open + self.is_marked)
        is_affected = np.zeros_like(is_trial, dtype=bool)
        for i in range(self.rows):
            for j in range(self.cols):
                is_affected[i, j] = self.count_neighbors(i, j, is_trial) > 0
        is_affected = is_affected * np.logical_not(is_trial)

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
        self.mine_freq = freq.reshape((self.rows, self.cols))
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

    def _neighborhood(self, i: int, j: int) -> Generator[Tuple[int, int], None, None]:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue  # do not count center
                if 0 <= i + di < self.rows and 0 <= j + dj < self.cols:
                    yield i + di, j + dj

    def count_neighbors(self, i: int, j: int, values: np.ndarray) -> int:
        return sum(
            int(values[i_n, j_n]) for i_n, j_n in self._neighborhood(i, j)
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
                        scaled_mine_prob = self.mine_freq[i, j] / np.max(self.mine_freq)
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
            scaled_prob_values = freq_values / np.max(self.mine_freq)

            print(ansi.modify('Mine probability:\n', codes=[ansi.TEXT.BOLD]))
            print(''.join([s * 6 for s in [self.mine_prob_colormap(sp) for sp in scaled_prob_values]]))
            print(''.join([f' {p:.2f} ' for p in mine_prob_values]))
