"""
NumPy practice: np.array and friends
------------------------------------
Each exercise is a function with a TODO. Replace `...` with your code,
then run:   python3 numpy_practice.py
The tests at the bottom tell you which ones pass. Solutions are at the very
end of the file (no peeking until you've had a proper go!).
"""
import numpy as np

# ---------------------------------------------------------------- Level 1: basics
def ex01_make_array():
    """Return a NumPy array of the numbers 3, 6, 9, 12."""
    return np.array([3, 6, 9, 12])

def ex02_shape_and_dtype():
    """Return a tuple (shape, dtype name) for a 3x4 array of zeros.
    Hint: np.zeros((rows, cols)) ; use .shape and .dtype.name"""
    z = np.zeros((3, 4))
    return (z.shape, z.dtype.name)

def ex03_ten_percent_raise(salaries):
    """Given an array of salaries, return them all increased by 10%. No loops!"""
    return salaries * 1.1

def ex04_third_column(grid):
    """grid is 2D. Return its third column (index 2) as a 1D array."""
    return grid[:, 2]

def ex05_last_row(grid):
    """Return the last row of a 2D grid."""
    return grid[-1]

# ---------------------------------------------------------------- Level 2: stats & masks
def ex06_weather_stats(temps):
    """Return (mean rounded to 1 dp, max, index of max)."""
    return (round(float(temps.mean()), 1), int(temps.max()), int(temps.argmax()))

def ex07_warm_days(temps):
    """Return only the temperatures strictly above 15."""
    return temps[temps > 15]

def ex08_count_warm_days(temps):
    """Return HOW MANY temperatures are strictly above 15 (an int)."""
    return int((temps > 15).sum())

def ex09_replace_outliers(values):
    """Return a copy where every value greater than 100 is replaced by 100.
    Hint: np.where(condition, if_true, if_false)  OR  copy + mask assignment."""
    return np.where(values > 100, 100, values)

def ex10_median_vs_mean(salaries):
    """Return (mean, median). Which is the more honest 'typical' salary?"""
    return (salaries.mean(), np.median(salaries))

# ---------------------------------------------------------------- Level 3: vector maths
def ex11_shopping_total(qty, price):
    """Return the total bill using the @ operator."""
    return qty @ price

def ex12_cosine(a, b):
    """Return cosine similarity between two 1D arrays, rounded to 3 dp."""
    return round(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))), 3)

def ex13_normalise_image(img):
    """img is uint8 (0..255). Return a float array scaled to 0.0..1.0."""
    return img / 255.0

def ex14_flip_vertically(img):
    """Return the image with rows reversed (upside down). Hint: slicing with -1 step."""
    return img[::-1]

def ex15_most_common(words):
    """words is an array of strings. Return the most common word.
    Hint: np.unique(..., return_counts=True) then argmax."""
    u, c = np.unique(words, return_counts=True)
    return u[c.argmax()]

# ---------------------------------------------------------------- Level 4: gotchas
def ex16_safe_copy(arr):
    """Return a NEW array equal to arr with its first element set to 999,
    WITHOUT changing the original arr."""
    new = arr.copy()
    new[0] = 999
    return new

def ex17_brighten(img):
    """img is uint8. Add 100 to every pixel but cap at 255 (no wrap-around!).
    Hint: convert to a bigger int type first, then np.clip, then back to uint8."""
    return np.clip(img.astype(np.int16) + 100, 0, 255).astype(np.uint8)

def ex18_top_n(values, n):
    """Return the n largest values, largest first. Hint: np.sort then slicing."""
    return np.sort(values)[::-1][:n]

# ---------------------------------------------------------------- Level 5: combining & broadcasting
def ex19_stack_rows(a, b):
    """a is 2x2, b is 1x2. Return a 3x2 array with b added as a new bottom row.
    Hint: np.vstack([...]) or np.concatenate([...], axis=0)."""
    return np.vstack([a, b])

def ex20_add_column(a, col):
    """a is 2x2, col is a 1D array of length 2. Return 2x3 with col as the new last column.
    Hint: col needs to become a column first -> col.reshape(-1, 1) or col[:, None]."""
    return np.hstack([a, col.reshape(-1, 1)])

def ex21_broadcast_row(a, b):
    """Add row vector b (shape 1x2) to EVERY row of a (2x2). One line, no loop."""
    return a + b

def ex22_centre_columns(a):
    """Subtract each column's mean from that column (so every column averages 0).
    Hint: a.mean(axis=0) is shape (2,), it broadcasts across rows."""
    return a - a.mean(axis=0)

def ex23_view_trap(a):
    """Take the first row of a as a slice, set its first element to 99,
    and return the ORIGINAL a. (Run it: does a change? That's the view trap.)"""
    row = a[0]
    row[0] = 99
    return a


# ================================================================ TESTS
def _run_tests():
    t = np.array([16, 16, 20, 17, 13, 13, 15])
    grid = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    img = np.array([[0, 128, 255], [64, 192, 32]], dtype=np.uint8)
    sal = np.array([65000, 72000, 80000, 95000, 120000, 70000])
    words = np.array(["python", "is", "fun", "python", "rocks", "python"])
    orig = np.array([1, 2, 3])

    checks = [
        ("ex01", lambda: np.array_equal(ex01_make_array(), [3, 6, 9, 12])),
        ("ex02", lambda: ex02_shape_and_dtype() == ((3, 4), "float64")),
        ("ex03", lambda: np.allclose(ex03_ten_percent_raise(np.array([100., 200.])), [110., 220.])),
        ("ex04", lambda: np.array_equal(ex04_third_column(grid), [3, 7, 11])),
        ("ex05", lambda: np.array_equal(ex05_last_row(grid), [9, 10, 11, 12])),
        ("ex06", lambda: ex06_weather_stats(t) == (15.7, 20, 2)),
        ("ex07", lambda: np.array_equal(ex07_warm_days(t), [16, 16, 20, 17])),
        ("ex08", lambda: ex08_count_warm_days(t) == 4),
        ("ex09", lambda: np.array_equal(ex09_replace_outliers(np.array([5, 150, 99, 101])), [5, 100, 99, 100])),
        ("ex10", lambda: ex10_median_vs_mean(sal) == (sal.mean(), 76000.0)),
        ("ex11", lambda: np.isclose(ex11_shopping_total(np.array([2, 1, 3]), np.array([4.5, 12.0, 2.25])), 27.75)),
        ("ex12", lambda: ex12_cosine(np.array([0.2, 0.9, 0.1]), np.array([0.3, 0.8, 0.2])) == 0.983),
        ("ex13", lambda: np.allclose(ex13_normalise_image(img), img / 255.0) and ex13_normalise_image(img).dtype == np.float64),
        ("ex14", lambda: np.array_equal(ex14_flip_vertically(img), img[::-1])),
        ("ex15", lambda: ex15_most_common(words) == "python"),
        ("ex16", lambda: np.array_equal(ex16_safe_copy(orig), [999, 2, 3]) and np.array_equal(orig, [1, 2, 3])),
        ("ex17", lambda: np.array_equal(ex17_brighten(img), [[100, 228, 255], [164, 255, 132]]) and ex17_brighten(img).dtype == np.uint8),
        ("ex18", lambda: np.array_equal(ex18_top_n(np.array([4, 9, 1, 7, 3]), 3), [9, 7, 4])),
        ("ex19", lambda: np.array_equal(ex19_stack_rows(np.array([[1,2],[3,4]]), np.array([[5,6]])), [[1,2],[3,4],[5,6]])),
        ("ex20", lambda: np.array_equal(ex20_add_column(np.array([[1,2],[3,4]]), np.array([9,9])), [[1,2,9],[3,4,9]])),
        ("ex21", lambda: np.array_equal(ex21_broadcast_row(np.array([[1,2],[3,4]]), np.array([[5,6]])), [[6,8],[8,10]])),
        ("ex22", lambda: np.allclose(ex22_centre_columns(np.array([[1,2],[3,4]])), [[-1,-1],[1,1]])),
        ("ex23", lambda: np.array_equal(ex23_view_trap(np.array([[1,2],[3,4]])), [[99,2],[3,4]])),
    ]
    passed = 0
    for name, fn in checks:
        try:
            ok = bool(fn())
        except Exception as e:
            ok = False
            print(f"  {name}: ERROR -> {type(e).__name__}: {e}")
        print(f"  {name}: {'PASS' if ok else 'fail'}")
        passed += ok
    print(f"\n{passed}/{len(checks)} passed")

if __name__ == "__main__":
    _run_tests()


# ================================================================ SOLUTIONS
# (Scroll down only when you're stuck or want to compare.)
#
# ex01: return np.array([3, 6, 9, 12])
# ex02: z = np.zeros((3, 4)); return (z.shape, z.dtype.name)
# ex03: return salaries * 1.1
# ex04: return grid[:, 2]
# ex05: return grid[-1]
# ex06: return (round(float(temps.mean()), 1), int(temps.max()), int(temps.argmax()))
# ex07: return temps[temps > 15]
# ex08: return int((temps > 15).sum())
# ex09: return np.where(values > 100, 100, values)
# ex10: return (salaries.mean(), np.median(salaries))
# ex11: return qty @ price
# ex12: return round(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))), 3)
# ex13: return img / 255.0
# ex14: return img[::-1]
# ex15: u, c = np.unique(words, return_counts=True); return u[c.argmax()]
# ex16: new = arr.copy(); new[0] = 999; return new
# ex17: return np.clip(img.astype(np.int16) + 100, 0, 255).astype(np.uint8)
# ex18: return np.sort(values)[::-1][:n]
# ex19: return np.vstack([a, b])
# ex20: return np.hstack([a, col.reshape(-1, 1)])
# ex21: return a + b
# ex22: return a - a.mean(axis=0)
# ex23: row = a[0]; row[0] = 99; return a
