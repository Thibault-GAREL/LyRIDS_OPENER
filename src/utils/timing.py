"""Helpers de mesure de latence + débit pour les benchmarks d'inférence.

On distingue :
    - `LatencyMeter`   : mesure individuelle (avec warmup), retourne p50/p95/p99/débit.
    - `time_function`  : décorateur léger pour mesurer une fonction unique.

Pourquoi pas juste `time.perf_counter()` partout : sur GPU, il faut un
`torch.cuda.synchronize()` AVANT le stop sinon on mesure la queue async, pas le
vrai calcul. Ce module l'encapsule.
"""
import statistics
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter


def _cuda_sync_if_available():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


@dataclass
class LatencyStats:
    n_samples: int
    total_seconds: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_per_s: float

    def as_dict(self) -> dict:
        return {'n_samples': self.n_samples,
                'total_seconds': round(self.total_seconds, 3),
                'p50_ms': round(self.p50_ms, 3),
                'p95_ms': round(self.p95_ms, 3),
                'p99_ms': round(self.p99_ms, 3),
                'throughput_per_s': round(self.throughput_per_s, 2)}


class LatencyMeter:
    """Accumule des mesures de latence puis calcule les stats.

    Usage :
        meter = LatencyMeter()
        meter.warmup(lambda: model(input_dummy), n=3)
        for x in inputs:
            with meter.measure():
                model(x)
        stats = meter.stats()
    """

    def __init__(self):
        self.times_s: list[float] = []

    def warmup(self, fn, n: int = 3) -> None:
        """Lance la fonction n fois pour amorcer le GPU (kernel compile, caches)."""
        for _ in range(n):
            try:
                fn()
            except Exception:
                pass
        _cuda_sync_if_available()

    @contextmanager
    def measure(self):
        _cuda_sync_if_available()
        t0 = perf_counter()
        try:
            yield
        finally:
            _cuda_sync_if_available()
            self.times_s.append(perf_counter() - t0)

    def add_seconds(self, dt: float) -> None:
        """Ajoute une mesure brute (utile si on a déjà mesuré avant)."""
        self.times_s.append(dt)

    def stats(self) -> LatencyStats:
        if not self.times_s:
            return LatencyStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        total = sum(self.times_s)
        sorted_ms = sorted(t * 1000 for t in self.times_s)
        n = len(sorted_ms)

        def percentile(p: float) -> float:
            if n == 1:
                return sorted_ms[0]
            k = (n - 1) * p / 100
            f = int(k)
            c = min(f + 1, n - 1)
            return sorted_ms[f] + (sorted_ms[c] - sorted_ms[f]) * (k - f)

        return LatencyStats(
            n_samples=n,
            total_seconds=total,
            p50_ms=statistics.median(sorted_ms),
            p95_ms=percentile(95),
            p99_ms=percentile(99),
            throughput_per_s=n / total if total > 0 else 0.0,
        )


@contextmanager
def time_function(label: str = 'fn'):
    """Petit context manager pour timer une opération unique (debug)."""
    _cuda_sync_if_available()
    t0 = perf_counter()
    try:
        yield
    finally:
        _cuda_sync_if_available()
        print(f"[{label}] {(perf_counter() - t0) * 1000:.2f} ms")
