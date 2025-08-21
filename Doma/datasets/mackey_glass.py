import numpy as np
from .base import TimeSeriesDataset

class MackeyGlassDataset(TimeSeriesDataset):
    # Presets from literature (based on Table 3 from @wringeReservoirComputingBenchmarks2025)
    PRESETS = {
        # ones below are from @jaegerEchoStateApproach
        "jaeger_tau17": {"length": 3000, "tau": 17, "beta": 0.2, "gamma": 0.1, "n": 10, "delta_t": 1},
        "jaeger_tau30": {"length": 3000, "tau": 30, "beta": 0.2, "gamma": 0.1, "n": 10, "delta_t": 1},
        "long_seq_tau17": {"length": 21000, "tau": 17, "beta": 0.2, "gamma": 0.1, "n": 10, "delta_t": 1},
        "long_seq_tau30": {"length": 21000, "tau": 30, "beta": 0.2, "gamma": 0.1, "n": 10, "delta_t": 1},
    }

    def __init__(self, preset=None, **kwargs):
        if preset:
            params = self.PRESETS[preset].copy()
        else:
            params = {}
        params.update(kwargs)

        length = params.pop("length", 2000)
        super().__init__(length)
        
        # Dataset parameters
        self.beta = params.get("beta", 0.2)
        self.gamma = params.get("gamma", 0.1)
        self.n = params.get("n", 10)
        self.tau = params.get("tau", 17)
        self.delta_t = params.get("delta_t", 1)


    def generate(self):
        np.random.seed(self.seed)
        N = self.length
        delay_steps = int(self.tau / self.delta_t)
        x = np.zeros(N + delay_steps)
        x[:delay_steps] = 1.2

        for t in range(delay_steps, N + delay_steps - 1):
            x_tau = x[t - delay_steps]
            dxdt = self.beta * x_tau / (1 + x_tau**self.n) - self.gamma * x[t]
            x[t + 1] = x[t] + dxdt * self.delta_t

        self.series = x[delay_steps:]
        return self.series
    
    def info(self):
        base_info = super().info()
        mg_params = {
            "beta": getattr(self, "beta", None),
            "gamma": getattr(self, "gamma", None),
            "n": getattr(self, "n", None),
            "tau": getattr(self, "tau", None),
            "delta_t": getattr(self, "delta_t", None)
        }
        base_info.update(mg_params)
        return base_info
