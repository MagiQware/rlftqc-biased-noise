"""Optional Weights & Biases logging for the shared compiled training loop."""

from contextlib import nullcontext
from threading import Lock

import jax
import jax.numpy as jax_numpy
import numpy

from rlftqc.agents import make_train


class EpisodeLogger:
    """Write completed episodes and rollout means, separately for each agent."""

    def __init__(self, run, configuration):
        self.run = run
        self.rollout_step_count = configuration["NUM_STEPS"]
        self.environment_count = configuration["NUM_ENVS"]
        # Host callbacks may execute concurrently for different vectorized agents.
        self.lock = Lock()
        self.pending_rollouts = {index: {} for index in range(configuration["NUM_AGENTS"])}
        self.next_update_indices = {index: 0 for index in range(configuration["NUM_AGENTS"])}
        for agent_index in range(configuration["NUM_AGENTS"]):
            prefix = f"agents/{agent_index}"
            run.define_metric(f"{prefix}/environment_steps")
            for metric_name in ("episode_reward", "episode_steps", "mean_episode_reward",
                                "mean_episode_steps", "completed_episodes"):
                run.define_metric(f"{prefix}/{metric_name}",
                                  step_metric=f"{prefix}/environment_steps")

    def __call__(self, agent_index, update_index, completed, rewards, lengths):
        """Ignore carried-forward wrapper values unless an episode actually ended."""
        completed = numpy.asarray(completed, dtype=bool)
        rewards = numpy.asarray(rewards)
        lengths = numpy.asarray(lengths)
        agent_index = int(agent_index)
        with self.lock:
            # Unordered device callbacks can arrive out of order. Buffer them so
            # each agent's chart always advances through rollouts chronologically.
            pending = self.pending_rollouts[agent_index]
            pending[int(update_index)] = (completed, rewards, lengths)
            next_update = self.next_update_indices[agent_index]
            while next_update in pending:
                self._log_rollout(agent_index, next_update, *pending.pop(next_update))
                next_update += 1
            self.next_update_indices[agent_index] = next_update

    def _log_rollout(self, agent_index, update_index, completed, rewards, lengths):
        """Write one chronological rollout while holding the callback lock."""
        prefix = f"agents/{agent_index}"
        rollout_start = update_index * self.rollout_step_count
        for step_index, environment_index in numpy.argwhere(completed):
            self.run.log({
                f"{prefix}/environment_steps": int(
                    (rollout_start + step_index + 1) * self.environment_count),
                f"{prefix}/episode_reward": float(rewards[step_index, environment_index]),
                f"{prefix}/episode_steps": int(lengths[step_index, environment_index]),
            })
        measurements = {
            f"{prefix}/environment_steps":
                (rollout_start + self.rollout_step_count) * self.environment_count,
            f"{prefix}/completed_episodes": int(completed.sum()),
        }
        # Empty rollouts still report progress, without misleading zero rewards.
        if completed.any():
            measurements[f"{prefix}/mean_episode_reward"] = float(rewards[completed].mean())
            measurements[f"{prefix}/mean_episode_steps"] = float(lengths[completed].mean())
        self.run.log(measurements)


def run_training(configuration, environment, seed, wandb_options=None):
    """Own an optional tracking run and drain callbacks before closing it.

    Pass a dictionary of ``wandb.init`` options to enable tracking, for example
    ``{"project": "rlftqc-biased-noise", "mode": "offline"}``.
    Omitting the dictionary does not import or initialize Weights & Biases.
    """
    run_context = nullcontext(None)
    if wandb_options is not None:
        try:
            import wandb
        except ImportError as error:
            raise ImportError(
                "Weights & Biases tracking requires wandb. Install it with: pip install wandb"
            ) from error
        options = dict(wandb_options)
        # Convert geometry arrays to ordinary lists for readable run configuration.
        recorded_configuration = {
            name: value.tolist() if hasattr(value, "tolist") else value
            for name, value in configuration.items()
        }
        recorded_configuration["random_seed"] = seed
        recorded_configuration.update(options.pop("config", {}))
        options.setdefault("project", "rlftqc-biased-noise")
        run_context = wandb.init(config=recorded_configuration, **options)

    with run_context as run:
        episode_callback = EpisodeLogger(run, configuration) if run is not None else None
        training_function = make_train(configuration, environment, episode_callback=episode_callback)
        compiled_training = jax.jit(jax.vmap(training_function))
        random_keys = jax.random.split(jax.random.PRNGKey(seed), configuration["NUM_AGENTS"])
        try:
            return jax.block_until_ready(compiled_training(
                random_keys, jax_numpy.arange(configuration["NUM_AGENTS"])))
        finally:
            # Device completion alone need not mean asynchronous host logging finished.
            jax.effects_barrier()
