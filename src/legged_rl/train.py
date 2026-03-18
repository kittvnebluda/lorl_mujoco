from logging import getLogger

import gymnasium as gym
import torch
from gymnasium.wrappers import RecordEpisodeStatistics
from numpy.random import randint
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from .callbacks import CustomMetricsCallback, HParamsCallback
from .utils import sanitize_for_hparams

logger = getLogger(__name__)

algos = {
    "ppo": {
        "class": PPO,
        "params": dict(
            policy="MlpPolicy",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            normalize_advantage=True,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
        ),
    },
    "sac": {
        "class": SAC,
        "params": dict(
            policy="MlpPolicy",
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=1_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            target_entropy="auto",
        ),
    },
}


def make_env():
    env = gym.make("UnitreeGo2Env-v0")
    env = RecordEpisodeStatistics(env)
    return env


def train(
    name: str,
    algo: str,
    total_timesteps: int,
    load_model: str = "",
    device: str = "cpu",
):
    if device == "xpu":
        assert torch.xpu.is_available(), "XPU not available"
        assert torch.xpu.device_count() > 0, "No XPU devices found"

    assert len(name) != 0, "name argument cannot be empty"
    assert len(algo) != 0, "algo argument cannot be empty"

    with make_env() as env:
        check_env(env)
        env_hparams = sanitize_for_hparams(getattr(env.unwrapped, "_saved_kwargs", {}))

    n_envs = 32
    random_seed = randint(0, 2**32 - 1)
    set_random_seed(random_seed)

    if load_model:
        model_name = load_model.replace(".zip", "")
    else:
        model_name = f"{name}_{algo}"

    model_save_name = f"{name}_{algo}"

    cfg = algos[algo]
    algo_cls = cfg["class"]
    params = cfg["params"].copy()

    hparams = {
        "algorithm": algo.upper(),
        "n_envs": n_envs,
        "seed": random_seed,
        "device": device,
        **env_hparams,
    }

    print(f"Using {algo.upper()}; Model name: {model_save_name}")

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints/",
        name_prefix=model_save_name,
    )

    vec_env = make_vec_env(
        make_env,
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        seed=random_seed,
    )
    vec_env = VecNormalize(
        vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0
    )

    params["device"] = device
    params["env"] = vec_env
    params["tensorboard_log"] = f"./tb_logs/mujoco_{algo}/"

    if load_model:
        print(f"Loading model {model_name}")
        model = algo_cls.load(model_name, **params)
    else:
        print(f"Creating new model {model_name}")
        model = algo_cls(**params)

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=[
            checkpoint_callback,
            CustomMetricsCallback(100),
            HParamsCallback(hparams),
        ],
    )
    model.save("./models/" + model_save_name)

    logger.info("Training finished")

    # Evaluate policy
    vec_env = model.get_env()
    if vec_env is None:
        logger.info("Vec Env is None")
        return

    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
    logger.info(f"Reward mean: {mean_reward}, std: {std_reward}")
