from typing import Any, Dict

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam


class HParamsCallback(BaseCallback):
    def __init__(self, hparams: dict, verbose: int = 0):
        super().__init__(verbose)
        self.hparams = hparams

    def _on_training_start(self) -> None:
        self.logger.record(
            "hparams",
            HParam(
                self.hparams,
                {
                    "episode/time": 0.0,
                    "step/vx_cmd_error": 10.0,
                    "step/vy_cmd_error": 10.0,
                    "step/wz_cmd_error": 10.0,
                    "step/body_height_cmd_error": 5.0,
                },
            ),
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


class AdaptiveCurriculumCallback(BaseCallback):
    """
    Adaptive curriculum that increases penalty weights (or other params)
    only when mean episode reward exceeds a threshold.

    Parameters example:
        {
            "pose_weight": {
                "start": 0.01,
                "set_method": "set_pose_unsimilarity_weight",
                "get_method": "get_pose_unsimilarity_weight",
                "log_name": "pose_weight"
            },
            "action_weight": {
                "start": 0.05,
                "set_method": "set_action_rate_weight",
                "get_method": "get_action_rate_weight",
                "log_name": "action_rate_weight"
            },
        }
    """

    def __init__(
        self,
        parameters: Dict[str, Dict[str, Any]],
        reward_threshold: float,
        increase_step: float,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._params = parameters
        self._rew_thresh = reward_threshold
        self._inc_step = increase_step

        for _, param in parameters.items():
            assert "start" in param.keys()
            assert "set_method" in param.keys()
            assert "get_method" in param.keys()
            assert "log_name" in param.keys()

            assert isinstance(param["start"], float)
            assert isinstance(param["set_method"], str)
            assert isinstance(param["get_method"], str)
            assert isinstance(param["log_name"], str)

    def _on_training_start(self) -> None:
        for _, cfg in self._params.items():
            cfg["current"] = cfg["start"]
            self.training_env.env_method(cfg["set_method"], cfg["current"])

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.model.ep_info_buffer is None or len(self.model.ep_info_buffer) == 0:
            return

        rew = sum(self.model.ep_info_buffer[-i]["r"] for i in range(1, 6)) / 5
        if rew > self._rew_thresh:
            for _, cfg in self._params.items():
                new = cfg["current"] + self._inc_step
                self.training_env.env_method(cfg["set_method"], new)
                cfg["current"] = self.training_env.env_method(cfg["get_method"])[0]

        for _, cfg in self._params.items():
            self.logger.record(f"curriculum/{cfg['log_name']}", cfg["current"])
