from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, Literal, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization


class ActorCriticDreamWaQ(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        cenet_cfg: dict,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticDreamWaQ.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs])
            )
        super().__init__()

        # Get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            num_actor_obs += obs[obs_group].shape[-1]
        for key in cenet_cfg["estimated_state_dims"]:
            num_actor_obs += cenet_cfg["estimated_state_dims"][key]
        num_actor_obs += cenet_cfg["latent_state_dim"]

        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            num_critic_obs += obs[obs_group].shape[-1]

        # Actor
        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            self.actor = MLP(num_actor_obs, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        print(f"Actor MLP: {self.actor}")

        # CENet
        self.cenet = CENet(obs, obs_groups, activation=activation, **cenet_cfg, **kwargs)
        print(f"CENet: {self.cenet}")

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            # Get the normalization dimension
            actor_obs_normalization_dim = 0
            for obs_group in obs_groups["policy"]:
                actor_obs_normalization_dim += obs[obs_group].shape[-1]

            self.actor_obs_normalizer = EmpiricalNormalization(actor_obs_normalization_dim)
            for key in cenet_cfg["estimated_state_dims"]:
                setattr(self, f"actor_{key}_normalizer", EmpiricalNormalization(cenet_cfg["estimated_state_dims"][key]))
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
            for key in cenet_cfg["estimated_state_dims"]:
                setattr(self, f"actor_{key}_normalizer", torch.nn.Identity())

        # Critic
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            if actor_obs_normalization:
                self.critic_obs_normalizer = self.actor_obs_normalizer
                for key in cenet_cfg["estimated_state_dims"]:
                    setattr(self, f"critic_{key}_normalizer", getattr(self, f"actor_{key}_normalizer"))
            else:
                # Get the normalization dimension
                critic_obs_normalization_dim = 0
                for obs_group in obs_groups["policy"]:
                    critic_obs_normalization_dim += obs[obs_group].shape[-1]

                self.critic_obs_normalizer = EmpiricalNormalization(critic_obs_normalization_dim)
                for key in cenet_cfg["estimated_state_dims"]:
                    setattr(self, f"critic_{key}_normalizer", EmpiricalNormalization(cenet_cfg["estimated_state_dims"][key]))
        else:
            self.critic_obs_normalizer = torch.nn.Identity()
            for key in cenet_cfg["estimated_state_dims"]:
                setattr(self, f"critic_{key}_normalizer", torch.nn.Identity())

        # Action noise
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, obs: torch.Tensor) -> None:
        if self.state_dependent_std:
            # Compute mean and standard deviation
            mean_and_std = self.actor(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            # Compute mean
            mean = self.actor(obs)
            # Compute standard deviation
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, bootstrap: bool, stage: Literal["rollout", "update"], **kwargs: dict[str, Any]) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            curr_obs, history_obs = self.get_actor_obs(obs, "all")
            curr_obs_normalized = self.actor_obs_normalizer(curr_obs)
            history_obs_normalized = self.actor_obs_normalizer(history_obs.reshape(-1, curr_obs.shape[-1])).reshape(curr_obs.shape[0], -1)
        encode_lin_vel, encode_context, context_mean, context_logvar = self.cenet.encode(history_obs_normalized)
        with torch.no_grad():
            if bootstrap:
                lin_vel_normalized = self.actor_lin_vel_normalizer(encode_lin_vel)
            else:
                lin_vel = obs["privileged"][:, :3]
                lin_vel_normalized = self.actor_lin_vel_normalizer(lin_vel)
        actor_obs = torch.cat((curr_obs_normalized, lin_vel_normalized.detach(), encode_context.detach()), dim=-1)
        self._update_distribution(actor_obs)
        if stage == "rollout":
            return self.distribution.sample()
        elif stage == "update":
            return encode_lin_vel, encode_context, context_mean, context_logvar
        else:
            raise ValueError(f"Unknown stage: {stage}. Please choose 'rollout' or 'update'")

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        curr_obs, history_obs = self.get_actor_obs(obs, "all")
        curr_obs_normalized = self.actor_obs_normalizer(curr_obs)
        history_obs_normalized = self.actor_obs_normalizer(history_obs.reshape(-1, curr_obs.shape[-1])).reshape(curr_obs.shape[0], -1)
        encode_lin_vel, _, context_mean, _ = self.cenet.encode(history_obs_normalized)
        lin_vel_normalized = self.actor_lin_vel_normalizer(encode_lin_vel)
        actor_obs = torch.cat((curr_obs_normalized, lin_vel_normalized.detach(), context_mean.detach()), dim=-1)
        if self.state_dependent_std:
            return self.actor(actor_obs)[..., 0, :]
        else:
            return self.actor(actor_obs)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        with torch.no_grad():
            obs_list = self.get_critic_obs(obs)
            curr_obs_normalized = self.critic_obs_normalizer(obs_list[0])
            lin_vel_normalized = self.critic_lin_vel_normalizer(obs_list[1][:, :3])
            external_force = obs_list[1][:, 3:]
            height_scan = obs_list[2]
            critic_obs = torch.cat((curr_obs_normalized, lin_vel_normalized, external_force, height_scan), dim=-1)
        return self.critic(critic_obs)

    def get_actor_obs(self, obs: TensorDict, mode: Literal["current", "history", "all"]) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        obs_concat = torch.cat(obs_list, dim=-1)
        curr_obs = obs_concat[:, -1, :]
        history_obs = obs_concat.reshape(obs_concat.shape[0], -1)
        if mode == "current":
            return curr_obs
        elif mode == "history":
            return history_obs
        elif mode == "all":
            return curr_obs, history_obs
        else:
            raise ValueError(f"Unknown mode: {mode}. Please choose 'current', 'history', or 'all'")

    def get_critic_obs(self, obs: TensorDict) -> list[torch.Tensor]:
        return list(obs[obs_group] for obs_group in self.obs_groups["critic"])

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            obs_list = self.get_critic_obs(obs)
            self.actor_obs_normalizer.update(obs_list[0])
            self.actor_lin_vel_normalizer.update(obs_list[1][:, :3])
        if self.critic_obs_normalization and not self.actor_obs_normalization:
            obs_list = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(obs_list[0])
            self.critic_lin_vel_normalizer.update(obs_list[1][:, :3])

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the actor-critic model.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        super().load_state_dict(state_dict, strict=strict)
        return True


class CENet(nn.Module):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        encoder_hidden_dims: tuple[int] | list[int] = [128],
        decoder_hidden_dims: tuple[int] | list[int] = [64, 128],
        encoder_head_hidden_dims: dict[str, tuple[int] | list[int] | None] = {
            "lin_vel": None,
            "context_mean": None,
            "context_logvar": None,
        },
        encoder_output_dim: int = 64,
        estimated_state_dims: dict[str, int] = {"lin_vel": 3},
        latent_state_dim: int = 16,
        activation: str = "elu",
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "CENet.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs])
            )
        super().__init__()

        # Get the input and output dimensions
        encoder_input_dim = 0
        for obs_group in obs_groups["policy"]:
            encoder_input_dim += (obs[obs_group].shape[-1] * obs[obs_group].shape[1])

        decoder_input_dim = latent_state_dim
        for key in estimated_state_dims:
            decoder_input_dim += estimated_state_dims[key]
        decoder_output_dim = 0
        for obs_group in obs_groups["policy"]:
            decoder_output_dim += obs[obs_group].shape[-1]

        # Encoder
        self.encoder = MLP(encoder_input_dim, encoder_output_dim, encoder_hidden_dims, activation, last_activation=activation)

        # Encoder heads
        for key in encoder_head_hidden_dims:
            if encoder_head_hidden_dims[key] is not None:
                setattr(self, f"encoder_head_{key}", MLP(encoder_output_dim, latent_state_dim if "context" in key else estimated_state_dims[key], encoder_head_hidden_dims[key], activation))
            else:
                setattr(self, f"encoder_head_{key}", nn.Linear(encoder_output_dim, latent_state_dim if "context" in key else estimated_state_dims[key]))

        # Decoder
        self.decoder = MLP(decoder_input_dim, decoder_output_dim, decoder_hidden_dims, activation)

    def encode(self, obs_hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feature = self.encoder(obs_hist)
        encode_lin_vel = self.encoder_head_lin_vel(feature)
        context_mean = self.encoder_head_context_mean(feature)
        context_logvar = self.encoder_head_context_logvar(feature)
        context_logvar_clipped = (0.5 * context_logvar).exp().clip(0.0, 5.0).square().log()
        encode_context = self.reparameterize(context_mean, context_logvar_clipped)
        return encode_lin_vel, encode_context, context_mean, context_logvar_clipped

    def decode(self, estimated_states: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat((estimated_states.detach(), context), dim=-1))

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        encode = mean + epsilon * std
        return encode
