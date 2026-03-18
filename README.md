# Study of reinforcement learning algorithms for control of quadruped robots in obstacle traversal

## Development

The python package is built with pybind11 and does not ship
with type hints / `.pyi` stub files. They have to generated:

```bash
pybind11-stubgen mujoco -o typings
```

And then with `pyrightconfig.json` LSP should work correctly.
