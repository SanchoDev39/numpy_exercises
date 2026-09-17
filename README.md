# NumPy Exercises

A personal practice repository for learning NumPy fundamentals, plus a small standalone Python warm-up script.

## Contents

### `numpy_indexing.py`
A self-contained set of 23 NumPy exercises, each implemented as a function with tests built into the same file. Run it directly to see which exercises pass:

```bash
python3 numpy_indexing.py
```

The exercises are organized into five levels of increasing difficulty:

- **Level 1 — Basics** (`ex01`–`ex05`): creating arrays, inspecting shape/dtype, basic arithmetic, and indexing rows/columns of 2D arrays.
- **Level 2 — Stats & masks** (`ex06`–`ex10`): computing mean/max/median, boolean masking to filter or count values, and replacing outliers with `np.where`.
- **Level 3 — Vector maths** (`ex11`–`ex15`): the `@` matrix-multiplication operator, cosine similarity, image normalization, flipping arrays, and finding the most common value with `np.unique`.
- **Level 4 — Gotchas** (`ex16`–`ex18`): the difference between array copies and views, avoiding integer overflow when doing arithmetic on `uint8` data, and sorting/slicing to get top-N values.
- **Level 5 — Combining & broadcasting** (`ex19`–`ex23`): stacking arrays together, adding new columns, NumPy's broadcasting rules, and a demonstration of the "view trap" (mutating a slice mutates the original array).

Each function's docstring describes the task, and solutions are included in a comment block at the bottom of the file for reference after attempting the exercises.

### `practice.py`
A tiny standalone script practicing basic Python control flow (a `while` loop printing multiples of 2).

## Status
All 23 NumPy exercises in `numpy_indexing.py` currently pass.
