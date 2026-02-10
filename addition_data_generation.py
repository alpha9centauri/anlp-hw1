# Generate a dataset of addition problems
import random
import os

def generate_addition_data(n):
    """
    Generate a dataset of n unique 4-digit addition problems.

    The function should:
    1) Randomly sample two 4-digit integers a and b.
        Recall: 4-digit integers are integenrs in the range of [1000, 9999]
    2) Compute their sum c = a + b
    3) Ensure uniqueness of addition pairs, treating (a, b) and (b, a) as identical
    4) Repeat until n unique examples are collected
    5) Return the data formatted as strings of the form "a+b=c"
    """
    min_val, max_val = 1000, 9999
    total_unique_pairs = (max_val - min_val + 1) * (max_val - min_val + 2) // 2  # combinations with replacement

    if n < 0:
        raise ValueError("n must be non-negative")
    if n > total_unique_pairs:
        raise ValueError(f"Requested n={n}, but max unique unordered pairs is {total_unique_pairs}")

    seen_pairs = set()
    data = []

    while len(data) < n:
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)

        # canonical unordered representation so (a,b) == (b,a)
        pair = (a, b) if a <= b else (b, a)

        if pair in seen_pairs:
            continue

        seen_pairs.add(pair)
        c = a + b
        data.append(f"{a}+{b}={c}")

    return data


def generate_dataset(n, filename, save_dir="data"):
    data = generate_addition_data(n)
    os.makedirs('data', exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'w') as f:
        f.write('\n'.join(data))
    print(f"{n} data points saved to {filepath}")
