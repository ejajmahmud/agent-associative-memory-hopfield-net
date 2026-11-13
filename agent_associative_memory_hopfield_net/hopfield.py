"""Modern Continuous Hopfield Network associative memory."""
import math
from typing import List, Optional, Tuple, Any

def dot(v1: List[float], v2: List[float]) -> float:
    return sum(a * b for a, b in zip(v1, v2))

class ModernHopfieldMemory:
    """Continuous associative memory with exponential storage capacity."""

    def __init__(self, beta: float = 2.0):
        self.beta = beta
        self.patterns: List[List[float]] = []
        self.labels: List[str] = []

    def store_pattern(self, label: str, vector: List[float]):
        self.labels.append(label)
        self.patterns.append(vector)

    def retrieve(self, query: List[float], max_iter: int = 3) -> Tuple[str, List[float]]:
        """Updates query state using modern Hopfield softmax update rule."""
        if not self.patterns:
            raise ValueError("No stored patterns")

        state = list(query)
        for _ in range(max_iter):
            # Compute logits: beta * (patterns @ state)
            logits = [self.beta * dot(p, state) for p in self.patterns]
            max_l = max(logits)
            exps = [math.exp(l - max_l) for l in logits]
            sum_exp = sum(exps)
            weights = [e / sum_exp for e in exps]

            # Reconstruct state: sum(weight_i * pattern_i)
            dim = len(state)
            new_state = [0.0] * dim
            for w, p in zip(weights, self.patterns):
                for d in range(dim):
                    new_state[d] += w * p[d]
            state = new_state

        # Find closest pattern label
        best_idx = max(range(len(self.patterns)), key=lambda i: dot(self.patterns[i], state))
        return self.labels[best_idx], state
