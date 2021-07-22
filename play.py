import sys
from dataclasses import dataclass
from string import ascii_letters, digits

from typing import Callable

from board import Board
import ansi


b = Board(10, 13, 30)
b.render()


# print(''.join(b.FREQ_COLORMAP))
# print(''.join([b.mine_prob_colormap(freq / 30) for freq in range(30)]))


@dataclass
class Action:
    letter: str
    func: Callable[[int, int], None]
    description: str

    def __str__(self):
        words_modified = []
        for word in self.description.split():
            if word[0].lower() == self.letter.lower():
                word = ansi.modify(self.letter.upper(), [ansi.TEXT.UNDER]) + word[1:]
            words_modified.append(word)
        return ' '.join(words_modified)


actions = [
    Action('O', b.open_cell, 'Open'),
    Action('M', b.mark_cell, 'Mark'),
    Action('U', b.unmark_cell, 'Unmark'),
    Action('N', b.open_neighbors, 'Open neighbors'),
    Action('C', lambda *_: None, 'Change cell'),
    Action('F', lambda *_: b.play_forward(), 'Fast forward'),
    Action('E', lambda *_: b.evaluate_probabilities(), 'Evaluate probabilities'),
    Action('I', lambda *_: b.info(), 'Info'),
    Action('Q', lambda *_: sys.exit(), 'Quit'),
]


while True:
    while True:
        coords_str = input('Coordinates (leave empty for non-positional actions) -> ')
        if coords_str == '':
            i, j = None, None
            break
        try:
            i_str = ''.join([ch for ch in coords_str if ch in digits])
            i = int(i_str)
            j = ''.join([ch for ch in coords_str if ch in ascii_letters])
            b._validate_index(i, column=False)
            b._validate_index(j, column=True)
            b.render(pointer=(i, j))
            break
        except ValueError as e:
            print(f'Problem parsing coordinates, try again or ":q" for exit (details: {e})')
    target_action_letter = input(f'Action ({" | ".join([str(a) for a in actions])}) -> ').upper()
    for action in actions:
        if action.letter == target_action_letter:
            action.func(i, j)
            break
    else:
        continue
    b.render()

    if b.game_over:
        print('\ngame over :(')
        break
    elif b.game_winned:
        print(ansi.modify('\ncongratulations!', codes=[ansi.TEXT.RED, ansi.TEXT.UNDER]))
        break
