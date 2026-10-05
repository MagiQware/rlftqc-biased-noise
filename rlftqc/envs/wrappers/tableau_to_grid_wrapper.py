import jax
import jax.numpy as jnp
from functools import partial
from gymnax.wrappers.purerl import GymnaxWrapper


class TableauToGridWrapper(GymnaxWrapper):
    def __init__(self, env, mapping, rows=5, cols=5):
        super().__init__(env)
        self.mapping = jnp.asarray(mapping, dtype=jnp.int32)
        self.rows, self.cols = rows, cols
        self.n = self.mapping.shape[0]
        self.row_idx, self.col_idx = self.mapping.T

    def transform(self, obs):
        tableau = obs["tableau"]

        n = self.n
        grid_X = jnp.zeros((self.rows, self.cols, n), dtype=tableau.dtype)
        grid_Z = jnp.zeros((self.rows, self.cols, n), dtype=tableau.dtype)

        grid_X = grid_X.at[self.row_idx, self.col_idx, :].set(tableau[:, :n].T)
        grid_Z = grid_Z.at[self.row_idx, self.col_idx, :].set(tableau[:, n:].T)

        return {
            "grid": jnp.concatenate([grid_X, grid_Z], axis=-1),  # (5, 5, 42)
            "phases": obs["phases"],                              # (21,)
            **({"max_diff": obs["max_diff"]} if "max_diff" in obs else {}),
        }

    # @partial(jax.jit, static_argnums=0)
    def reset(self, key, params=None):
        tableau, state = self._env.reset(key, params)
        return self.transform(tableau), state

    # @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action, params=None):
        tableau, state, reward, terminated, truncated, info = (
            self._env.step(key, state, action, params)
        )

        info = {
            **info,
            "final_observation": self.transform(info["final_observation"]),
        }
        return (
            self.transform(tableau),
            state,
            reward,
            terminated,
            truncated,
            info,
        )