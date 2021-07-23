import sys
from dataclasses import dataclass
from string import ascii_letters, digits

from typing import Callable, Tuple

from board import Board
import ansi


b = Board(rows=15, cols=18, mines=65)
b.render()


@dataclass
class Action:
    letter: str
    func: Callable[[int, int], None]
    description: str
    is_positional: bool

    def __str__(self):
        words_modified = []
        for word in self.description.split():
            if word[0].lower() == self.letter.lower():
                word = ansi.modify(self.letter.upper(), [ansi.TEXT.UNDER]) + word[1:]
            words_modified.append(word)
        return ' '.join(words_modified)


actions = [
    Action('O', b.open_cell, 'Open', True),
    Action('M', b.mark_cell, 'Mark', True),
    Action('U', b.unmark_cell, 'Unmark', True),
    Action('N', b.open_neighbors, 'Open neighbors', True),
    Action('C', lambda *_: None, 'Change cell', False),
    Action('F', lambda *_: b.play_forward(), 'Fast forward', False),
    Action('E', lambda *_: b.evaluate_probabilities(), 'Evaluate probabilities', False),
    Action('I', lambda *_: b.info(), 'Info', False),
    Action('P', lambda *_: b.render_as_plot(), 'Plot', False),
    Action('Q', lambda *_: sys.exit(), 'Quit', False),
]


def input_coordinates() -> Tuple[int, int]:
    while True:
        coords_str = input('Coordinates (empty for non-positional actions) -> ')
        if coords_str == '':
            # non-positional action expected!
            return None, None
        try:
            i_str = ''.join([ch for ch in coords_str if ch in digits])
            i = int(i_str)
            j = ''.join([ch for ch in coords_str if ch in ascii_letters])
            b._validate_index(i, column=False)
            b._validate_index(j, column=True)
            b.render(pointer=(i, j))
            return i, j
        except ValueError as e:
            print(f'Can\'t coordinates, try again or leave empty (details: {e})')


while True:
    i, j = input_coordinates()

    target_action_letter = input(f'Action ({" | ".join([str(a) for a in actions])}) -> ').upper()
    for action in actions:
        if action.letter == target_action_letter:
            action.func(i, j)
            break
    else:
        continue

    b.render()

    if b.game_over:
        print(ansi.modify('\n\tgame over!', codes=[ansi.TEXT.RED, ansi.TEXT.UNDER]))
        break
    elif b.game_winned:
        print(ansi.modify('\ncongratulations!', codes=[ansi.TEXT.RED, ansi.TEXT.UNDER]))
        break
