import sys
from dataclasses import dataclass
from string import ascii_letters, digits

from typing import Callable, List

from board import Board
import ansi


@dataclass
class Action:
    letter: str
    func: Callable[..., None]
    description: str
    is_positional: bool

    def __str__(self):
        words_modified = []
        for word in self.description.split():
            if word[0].lower() == self.letter.lower():
                word = ansi.modify(self.letter.upper(), [ansi.TEXT.UNDER]) + word[1:]
            words_modified.append(word)
        return " ".join(words_modified)

    def __call__(self, *args):
        return self.func(*args)


@dataclass
class Coordinates:
    i: int
    j: str


def print_game_msg(s: str):
    print(ansi.modify(s, [ansi.TEXT.RED, ansi.TEXT.BOLD]))


def input_coordinates(b: Board) -> List[Coordinates]:
    while True:
        coords_str = input("Coordinates (empty for non-positional actions) -> ")
        coords_list = []
        for coord in coords_str.split():
            try:
                i_str = "".join([ch for ch in coord if ch in digits])
                i = int(i_str)
                j = "".join([ch for ch in coord if ch in ascii_letters])
                b._validate_index(i, column=False)
                b._validate_index(j, column=True)
                coords_list.append(Coordinates(i, j))
            except ValueError as e:
                print_game_msg(
                    f'Can\'t parse coordinates from "{coord}", try again or leave empty (details: {e})'
                )
                break
        else:
            break
    return coords_list


if __name__ == "__main__":
    b = Board(rows=15, cols=18, mines=60)
    actions = [
        Action("O", b.open_cell, "Open", True),
        Action("M", b.mark_cell, "Mark", True),
        Action("U", b.unmark_cell, "Unmark", True),
        Action("N", b.open_neighbors, "Open neighbors", True),
        Action("C", lambda: None, "Change cell", False),
        Action("F", lambda: b.fast_forward(), "Fast forward", False),
        Action(
            "E", lambda: b.evaluate_probabilities(), "Evaluate probabilities", False
        ),
        Action("I", lambda: b.info(), "Info", False),
        Action("P", lambda: b.render_as_plot(), "Plot", False),
        Action("Q", lambda: sys.exit(), "Quit", False),
    ]
    actions_by_letter = {a.letter: a for a in actions}

    b.render()
    while True:
        coords = input_coordinates(b)
        b.set_pointers([(c.i, c.j) for c in coords])
        if coords:
            b.render()

        while True:
            target_action_letter = input(
                f'Action ({" | ".join([str(a) for a in actions])}) -> '
            ).upper()
            action = actions_by_letter.get(target_action_letter, None)
            if action is not None:
                break
            else:
                print_game_msg(f"Invalid action letter")

        if action.is_positional:
            if not coords:
                print_game_msg(
                    f"{action.description} is positional action, provide coodinates!"
                )
                continue
            else:
                for coord in coords:
                    action(coord.i, coord.j)
        else:
            action()

        b.set_pointers([])
        b.render()

        if b.game_over:
            print(
                ansi.modify(
                    "\n\tgame over!\n", codes=[ansi.TEXT.YELLOW, ansi.TEXT.UNDER]
                )
            )
            break
        elif b.game_winned:
            print(
                ansi.modify(
                    "\n\tcongratulations!\n", codes=[ansi.TEXT.YELLOW, ansi.TEXT.UNDER]
                )
            )
            break
