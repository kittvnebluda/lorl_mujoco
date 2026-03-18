from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam


class HParamsCallback(BaseCallback):
    def __init__(self, hparams: dict, verbose: int = 0):
        super().__init__(verbose)
        self.hparams = hparams

    def _on_training_start(self) -> None:
        self.logger.record(
            "hparams",
            HParam(self.hparams, {"episode/time": 0.0}),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True


class CustomMetricsCallback(BaseCallback):
    def __init__(self, log_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq != 0:
            return True

        try:
            all_logs = self.training_env.get_attr("tb_logs")
        except AttributeError:
            return True

        if not all_logs:
            return True

        # Average across environments
        combined = {}
        for logs in all_logs:
            for k, v in logs.items():
                combined[k] = combined.get(k, 0.0) + float(v)

        n = len(all_logs)
        for k, total in combined.items():
            self.logger.record(k, total / n)

        self.training_env.env_method("clear_logs")

        return True
