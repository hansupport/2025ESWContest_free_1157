# core/smooth.py  (Python 3.6 호환)
from collections import Counter
from typing import Optional, Tuple, List
import numpy as np


class ProbSmoother:
    """
    최근 N 프레임의 top-1 후보를 수집해 다수결(+평균 확률)로 확정.
    """
    def __init__(self, window=3, min_votes=2):
        self.window = int(window)
        self.min_votes = int(min_votes)
        self.buf = []  # type: List[Tuple[str, float]]  # [(label, prob)]

    def push(self, label, prob):
        self.buf.append((str(label), float(prob)))
        if len(self.buf) > self.window:
            self.buf.pop(0)

    def status(self):
        if not self.buf:
            return None, 0, 0.0
        cnt = Counter([lab for lab, _ in self.buf])
        top_lab, votes = cnt.most_common(1)[0]
        avg_p = float(np.mean([p for lab, p in self.buf if lab == top_lab]))
        return top_lab, votes, avg_p

    def maybe_decide(self, threshold=0.40):
        # type: (float) -> Optional[Tuple[str, float]]
        if len(self.buf) < self.window:
            return None
        lab, votes, avg_p = self.status()
        if votes >= self.min_votes and avg_p >= threshold:
            decided = (lab, avg_p)
            self.buf.clear()
            return decided
        return None
