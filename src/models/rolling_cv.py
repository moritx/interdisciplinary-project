"""
Rolling-origin (expanding window) cross-validation utilities shared by all
models in this project - AR baseline, Lasso, Random Forest, neural net -
so every model is evaluated on exactly the same train/test splits and the
resulting forecast errors are directly comparable (required for the
Diebold-Mariano test later).
"""
from typing import Iterator, Tuple

import pandas as pd


def expanding_window_splits(
    index: pd.PeriodIndex, min_train_size: int
) -> Iterator[Tuple[pd.PeriodIndex, pd.Period]]:
    """Yield (train_index, test_period) pairs for one-step-ahead expanding-window CV.

    The first split trains on the first `min_train_size` observations and
    tests on the next one; each subsequent split adds one more observation
    to the training set. This mimics how new data actually arrives quarter
    by quarter in a real nowcasting deployment (an expanding, not sliding,
    window - all past data stays available).
    """
    n = len(index)
    for i in range(min_train_size, n):
        train_idx = index[:i]
        test_period = index[i]
        yield train_idx, test_period
