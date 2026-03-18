from gymnasium.envs.registration import register

register(
    id="UnitreeGo2Env-v0",
    entry_point="legged_rl.envs:UnitreeGo2Env",
)
