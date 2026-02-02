import gymnasium as gym
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv


class HIMLocoVecEnvWrapper(VecEnv):
    """Wraps around Isaac Lab environment for the HIMLoco RSL-RL implementation

    .. caution::
        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.

    Reference:
        https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/env/vec_env.py
    """

    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv, train_cfg: dict):
        """Initializes the wrapper.

        Args:
            env: The environment to wrap around.
            train_cfg: The training configuration.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """

        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )

        # initialize the wrapper
        self.env = env
        self.clip_actions = train_cfg.get("clip_actions", None)
        self.only_positive_rewards = train_cfg.get("only_positive_rewards", False)
        self.obs_groups = train_cfg.get("obs_groups")

        # store information required by HimLoco RSL-RL
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        # obtain dimensions of the environment
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)

        # modify the action space to the clip range
        self._modify_action_space()

        # obtain single-step observation dimensions from observation manager
        # These are the dimensions of observations returned by the environment at each step
        self.num_one_step_obs = self.unwrapped.observation_manager.group_obs_dim["proprioception_with_noise"][0]
        # Set history lengths
        self.history_length = train_cfg.get("history_length", 5)
        # Calculate total observation dimensions
        # history_length=0 means only current step (num_obs = num_one_step_obs)
        # history_length=1 means current + 1 past step (num_obs = num_one_step_obs * 2)
        self.num_obs = self.num_one_step_obs * (self.history_length + 1)
        # History buffer for policy observations
        self.obs_history_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float, device=self.device)

        # Initialize privileged observations related attributes
        self.num_one_step_privileged_obs = 0
        for group_name in train_cfg.get("obs_groups").get("critic"):
            self.num_one_step_privileged_obs += self.unwrapped.observation_manager.group_obs_dim[group_name][0]
        self.privileged_history_length = train_cfg.get("privileged_history_length", 0)
        self.num_privileged_obs = self.num_one_step_privileged_obs * (self.privileged_history_length + 1)
        self.privileged_obs_history_buf = torch.zeros(self.num_envs, self.num_privileged_obs, dtype=torch.float, device=self.device)

        # Buffers for terminated environments (HimLoco-specific)
        self.termination_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self.termination_privileged_obs = torch.empty(0, self.num_privileged_obs, dtype=torch.float, device=self.device)

        # reset at the start since the HimLoco runner does not call reset
        self.env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    def __getattr__(self, name):
        """Forward all other attribute access to wrapped environment."""
        return getattr(self.env, name)

    """
    Properties
    """

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[TensorDict, dict]:  # noqa: D102
        # reset the environment
        obs_dict, extras = self.env.reset()
        return TensorDict(obs_dict, batch_size=[self.num_envs]), extras

    def get_observations(self) -> torch.Tensor:
        """Returns the current observations history of the environment."""
        return self.obs_history_buf

    def get_privileged_observations(self) -> torch.Tensor:
        """Returns the current privileged observations history of the environment."""
        return self.privileged_obs_history_buf

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict, torch.Tensor, torch.Tensor,]:
        # clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # record step information
        obs_dict, obs_dict_before_reset, rew, terminated, truncated, extras = self.env.step(actions)
        # clip rewards if only_positive_rewards is True
        if self.only_positive_rewards:
            rew = torch.clamp(rew, min=0.0)
        # compute dones for compatibility with HIMLoco RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # track which environments terminated this step (HIMLoco-specific)
        self.termination_ids = dones.nonzero(as_tuple=False).flatten()

        # extract policy observations (single-step observations)
        current_obs = obs_dict["proprioception_with_noise"]
        # update policy observations history buffer
        # environment returns single-step obs, wrapper does history stacking
        # [new_single_step_obs, old_obs_history[:-num_one_step_obs]]
        if self.history_length > 0:
            self.obs_history_buf = torch.cat((current_obs, self.obs_history_buf[:, :-self.num_one_step_obs]), dim=-1)
            self.obs_history_buf[self.termination_ids] = current_obs[self.termination_ids].repeat(1, self.history_length + 1)
        else:
            self.obs_history_buf = current_obs

        # extract privileged observations and termination observations (single-step observations)
        current_privileged_obs_list = []
        termination_obs_list = []
        for group_name in self.obs_groups.get("critic"):
            current_privileged_obs_list.append(obs_dict[group_name])
            termination_obs_list.append(obs_dict_before_reset[group_name])
        current_privileged_obs = torch.cat(current_privileged_obs_list, dim=-1)
        termination_obs = torch.cat(termination_obs_list, dim=-1)
        # update privileged observations history buffer if available
        if self.privileged_history_length > 0:
            self.privileged_obs_history_buf = torch.cat((current_privileged_obs, self.privileged_obs_history_buf[:, :-self.num_one_step_privileged_obs]), dim=-1)
            self.privileged_obs_history_buf[self.termination_ids] = current_privileged_obs[self.termination_ids].repeat(1, self.privileged_history_length + 1)
        else:
            self.privileged_obs_history_buf = current_privileged_obs

        # extract termination observations
        self.termination_privileged_obs = termination_obs[self.termination_ids]

        # check for NaN/Inf in observations and rewards before returning
        if torch.isnan(self.privileged_obs_history_buf).any() or torch.isinf(self.privileged_obs_history_buf).any():
            raise ValueError("NaN or Inf detected in privileged_obs_history_buf")
        if torch.isnan(rew).any() or torch.isinf(rew).any():
            raise ValueError("NaN or Inf detected in rew")

        return self.obs_history_buf, self.privileged_obs_history_buf, rew, dones, extras, self.termination_ids, self.termination_privileged_obs

    def close(self):  # noqa: D102
        return self.env.close()

    """
    Helper functions
    """

    def _modify_action_space(self):
        """Modifies the action space to the clip range."""
        if self.clip_actions is None:
            return

        # modify the action space to the clip range
        # note: this is only possible for the box action space. we need to change it in the future for other
        #   action spaces.
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )
