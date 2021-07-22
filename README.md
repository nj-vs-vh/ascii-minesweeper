# ascii-minesweeper

A simple ASCII-graphics implementation of the minesweeper with 'assist mode'.

Controls from keyboard.

Assist mode:
 * simple: press 'F' to open open all obvious cells (i.e. when 2-cell has exactly 2 cells around)
 * probabilistic: press 'E' to brute-force all possible mine locations with respect to constrains and calculate probabilities for each unopened cell. Assumes that all marked mines are correct. Works for sufficiently small amount of mines and unopened cells left.


![main_view](screenshots/main.png)

![assist_view](screenshots/assist.png)


## Installation and usage

```bash
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
python play.py
```

To change field size and amount of mines edit values in play.py directly.
