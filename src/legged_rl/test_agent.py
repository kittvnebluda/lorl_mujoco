from time import sleep, time
import gymnasium as gym

from numpy import mean

from .train import algos


def test_agent(model_path: str, algo: str | None):
    assert model_path, "Invalid model argument"

    with gym.make("UnitreeGo2Env-v0", render_mode="human") as env:
        algo_name = model_path.split("_")[-1].replace(".zip", "")
        if algo is None:
            cfg = algos[algo_name]
        else:
            cfg = algos[algo]
        algo_cls = cfg["class"]
        params = cfg["params"].copy()
        params["env"] = env

        model = algo_cls.load(model_path, **params)

        vec_env = model.get_env()
        if vec_env is None:
            raise RuntimeError("VecEnv is None")

        obs = vec_env.reset()
        start_time = time()
        time_of_lifes = []

        try:
            while 1:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, info = vec_env.step(action)

                vec_env.env_method("print_debug")

                if dones[0]:
                    time_of_lifes.append(time() - start_time)
                    start_time = time()

                sleep(0.02)

        except KeyboardInterrupt:
            print()
            pass

        if time_of_lifes:
            print(f"Min time of life: {min(time_of_lifes):2f}s")
            print(f"Avg time of life: {mean(time_of_lifes):.2f}s")
            print(f"Max time of life: {max(time_of_lifes):2f}s")
