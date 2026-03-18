from typing import Optional

import typer

from .random_agent import random_agent
from .test_agent import test_agent
from .train import algos
from .train import train as train_policy

app = typer.Typer(
    name="legged-rl",
    help="Training and running agents for legged robots (Unitree Go2 etc.)",
    add_completion=True,
)


@app.command()
def train(
    name: str = typer.Option("go2", "--name", "-n", help="Name for model file"),
    algo: str = typer.Option(
        "ppo", "--algo", "-a", help=f"Algorithm: {list(algos.keys())}"
    ),
    total_timesteps: int = typer.Option(50000, "--total-timesteps", "-t"),
    load_model: str = typer.Option(
        "", "--load-model", "-l", help="Name of the file to load"
    ),
    device: str = typer.Option("cpu", "--device", "-d"),
):
    train_policy(
        name=name,
        algo=algo,
        total_timesteps=total_timesteps,
        load_model=load_model,
        device=device,
    )


@app.command()
def run(
    model: Optional[str] = typer.Argument(
        None, help="Path to model file (for model subcommand)"
    ),
    algo: Optional[str] = typer.Option(None, "--algo", "-a"),
):
    if model is not None:
        test_agent(model_path=model, algo=algo)
    else:
        random_agent()


if __name__ == "__main__":
    app()
