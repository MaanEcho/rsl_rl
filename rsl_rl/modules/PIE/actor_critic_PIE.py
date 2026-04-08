from __future__ import annotations

import copy
import torch
import torch.nn as nn
from pathlib import Path
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, Literal, NoReturn

from rsl_rl.networks import CNN, MLP, EmpiricalNormalization, HiddenState, Memory
from rsl_rl.utils import optimize_onnx_model, unpad_trajectories


class ActorCriticPIE(nn.Module):
    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        estimator_cfg: dict,
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
                "ActorCriticPIE.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )
        super().__init__()

        # Get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            if obs_group == "exteroception":
                continue
            num_actor_obs += obs[obs_group].shape[-1]
        for key in estimator_cfg["estimated_state_dims"]:
            num_actor_obs += estimator_cfg["estimated_state_dims"][key]
        num_actor_obs += estimator_cfg["height_scan_estimation_vector_dim"]
        num_actor_obs += estimator_cfg["latent_vector_dim"]

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

        # Estimator
        self.estimator = Estimator(obs, obs_groups, activation=activation, **estimator_cfg, **kwargs)
        print(f"Estimator: {self.estimator}")

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            # Get the normalization dimension
            actor_obs_normalization_dim = 0
            for obs_group in obs_groups["policy"]:
                if obs_group == "exteroception":
                    continue
                actor_obs_normalization_dim += obs[obs_group].shape[-1]

            self.actor_obs_normalizer = EmpiricalNormalization(actor_obs_normalization_dim)
            for key in estimator_cfg["estimated_state_dims"]:
                setattr(self, f"actor_{key}_normalizer", EmpiricalNormalization(estimator_cfg["estimated_state_dims"][key]))
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
            for key in estimator_cfg["estimated_state_dims"]:
                setattr(self, f"actor_{key}_normalizer", torch.nn.Identity())

        # Critic
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            if actor_obs_normalization:
                self.critic_obs_normalizer = self.actor_obs_normalizer
                for key in estimator_cfg["estimated_state_dims"]:
                    setattr(self, f"critic_{key}_normalizer", getattr(self, f"actor_{key}_normalizer"))
            else:
                # Get the normalization dimension
                critic_obs_normalization_dim = 0
                for obs_group in obs_groups["policy"]:
                    if obs_group == "exteroception":
                        continue
                    critic_obs_normalization_dim += obs[obs_group].shape[-1]

                self.critic_obs_normalizer = EmpiricalNormalization(critic_obs_normalization_dim)
                for key in estimator_cfg["estimated_state_dims"]:
                    setattr(self, f"critic_{key}_normalizer", EmpiricalNormalization(estimator_cfg["estimated_state_dims"][key]))
        else:
            self.critic_obs_normalizer = torch.nn.Identity()
            for key in estimator_cfg["estimated_state_dims"]:
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

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.estimator.rnn.reset(dones)

    def forward(self) -> NoReturn:
        raise NotImplementedError

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

    def act(self, obs: TensorDict, stage: Literal["rollout", "update"], masks: torch.Tensor | None = None, hidden_state: HiddenState = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            curr_obs, history_obs, depth_images = self.get_actor_obs(obs)
            curr_obs_normalized = self.actor_obs_normalizer(curr_obs)
            history_obs_normalized = self.actor_obs_normalizer(history_obs).flatten(-2)
        encode_lin_vel, encode_feet_clearances, z_m, z, z_mean, z_logvar = self.estimator.encode(history_obs_normalized, depth_images, masks, hidden_state)
        with torch.no_grad():
            lin_vel_normalized = self.actor_lin_vel_normalizer(encode_lin_vel)
            feet_clearances_normalized = self.actor_feet_clearances_normalizer(encode_feet_clearances)
        if masks is not None:
            curr_obs_normalized = unpad_trajectories(curr_obs_normalized, masks)
        actor_input = torch.cat((curr_obs_normalized, lin_vel_normalized.detach(), feet_clearances_normalized.detach(), z_m.detach(), z.detach()), dim=-1)
        self._update_distribution(actor_input)
        if stage == "rollout":
            return self.distribution.sample()
        elif stage == "update":
            return encode_lin_vel, encode_feet_clearances, z_m, z, z_mean, z_logvar
        else:
            raise ValueError(f"Unknown stage: {stage}. Please choose 'rollout' or 'update'")

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        curr_obs, history_obs, depth_images = self.get_actor_obs(obs)
        curr_obs_normalized = self.actor_obs_normalizer(curr_obs)
        history_obs_normalized = self.actor_obs_normalizer(history_obs).flatten(-2)
        encode_lin_vel, encode_feet_clearances, z_m, _, z_mean, _ = self.estimator.encode(history_obs_normalized, depth_images)
        lin_vel_normalized = self.actor_lin_vel_normalizer(encode_lin_vel)
        feet_clearances_normalized = self.actor_feet_clearances_normalizer(encode_feet_clearances)
        actor_input = torch.cat((curr_obs_normalized, lin_vel_normalized.detach(), feet_clearances_normalized.detach(), z_m.detach(), z_mean.detach()), dim=-1)
        if self.state_dependent_std:
            return self.actor(actor_input)[..., 0, :]
        else:
            return self.actor(actor_input)

    def evaluate(self, obs: TensorDict, masks: torch.Tensor | None = None) -> torch.Tensor:
        with torch.no_grad():
            obs_list = self.get_critic_obs(obs)
            if masks is not None:
                for i in range(len(obs_list)):
                    obs_list[i] = unpad_trajectories(obs_list[i], masks)
            curr_obs_normalized = self.critic_obs_normalizer(obs_list[0])
            lin_vel_normalized = self.critic_lin_vel_normalizer(obs_list[1][..., :3])
            feet_clearances_normalized = self.critic_feet_clearances_normalizer(obs_list[1][..., 3:7])
            height_scan = obs_list[1][..., 7:]
            critic_input = torch.cat((curr_obs_normalized, lin_vel_normalized, feet_clearances_normalized, height_scan), dim=-1)
        return self.critic(critic_input)

    def get_actor_obs(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        curr_obs = obs_list[0][..., -1, :].clone()
        history_obs = obs_list[0].clone()
        depth_images = obs_list[1].clone().squeeze(-1)
        return curr_obs, history_obs, depth_images

    def get_critic_obs(self, obs: TensorDict) -> list[torch.Tensor]:
        return list(obs[obs_group] for obs_group in self.obs_groups["critic"])

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self) -> HiddenState:
        return self.estimator.rnn.hidden_state

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            obs_list = self.get_critic_obs(obs)
            self.actor_obs_normalizer.update(obs_list[0])
            self.actor_lin_vel_normalizer.update(obs_list[1][:, :3])
            self.actor_feet_clearances_normalizer.update(obs_list[1][:, 3:7])
        if self.critic_obs_normalization and not self.actor_obs_normalization:
            obs_list = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(obs_list[0])
            self.critic_lin_vel_normalizer.update(obs_list[1][:, :3])
            self.critic_feet_clearances_normalizer.update(obs_list[1][:, 3:7])

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

    def export_policy(self, obs: TensorDict, path: Path) -> None:
        policy = InferenceWrapper(self)
        self._export_depth_encoder_as_jit(self.estimator.depth_encoder, path)
        self._export_policy_as_jit(policy, path)
        self._export_depth_encoder_as_onnx(obs, self.estimator.depth_encoder, path)
        self._export_policy_as_onnx(obs, policy, path)

    def _export_depth_encoder_as_jit(self, model: nn.Module, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = path.stem.split("_")[-1]
        export_path = path.with_name(f"depth_encoder_{checkpoint}_PIE.pt")
        with torch.no_grad():
            depth_encoder = copy.deepcopy(model).to("cpu")
            depth_encoder.eval()
            for p in depth_encoder.parameters():
                p.requires_grad_(False)
            depth_encoder_scripted = torch.jit.script(depth_encoder)
            depth_encoder_frozen = torch.jit.freeze(depth_encoder_scripted)
            depth_encoder_optimized = torch.jit.optimize_for_inference(depth_encoder_frozen)
            depth_encoder_optimized.save(export_path)

    def _export_policy_as_jit(self, model: nn.Module, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = path.stem.split("_")[-1]
        export_path = path.with_name(f"policy_{checkpoint}_PIE.pt")
        with torch.no_grad():
            policy = copy.deepcopy(model).to("cpu")
            policy.eval()
            for p in policy.parameters():
                p.requires_grad_(False)
            policy_scripted = torch.jit.script(policy)
            policy_frozen = torch.jit.freeze(policy_scripted)
            policy_optimized = torch.jit.optimize_for_inference(policy_frozen)
            policy_optimized.save(export_path)

    def _export_depth_encoder_as_onnx(self, obs: TensorDict, model: nn.Module, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = path.stem.split("_")[-1]
        export_path = path.with_name(f"depth_encoder_{checkpoint}_PIE.onnx")
        with torch.no_grad():
            depth_encoder = copy.deepcopy(model).to("cpu")
            depth_encoder.eval()
            for p in depth_encoder.parameters():
                p.requires_grad_(False)
            _, _, depth_images = self.get_actor_obs(obs)
            depth_images = depth_images.to("cpu")
            torch.onnx.export(
                depth_encoder,
                depth_images,
                export_path,
                input_names=["depth_images"],
                output_names=["depth_features"],
                opset_version=17,
                optimize=False,  # Optimize later
                export_params=True,
                dynamic_axes={
                    "depth_images": {0: "batch_size"},
                    "depth_features": {0: "batch_size"},
                },
                do_constant_folding=True,
            )

            import onnx

            onnx.checker.check_model(export_path, full_check=True)
            onnx_model = onnx.load(export_path)
            optimize_onnx_model(onnx_model, export_path, verbose=True)
            onnx.checker.check_model(export_path, full_check=True)

    def _export_policy_as_onnx(self, obs: TensorDict, model: nn.Module, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = path.stem.split("_")[-1]
        export_path = path.with_name(f"policy_{checkpoint}_PIE.onnx")
        with torch.no_grad():
            policy = copy.deepcopy(model).to("cpu")
            policy.eval()
            for p in policy.parameters():
                p.requires_grad_(False)
            obs, obs_hist, _ = self.get_actor_obs(obs)
            c_out = self.estimator.depth_encoder.output_channels
            if c_out is not None:
                h_out, w_out = self.estimator.depth_encoder.output_dim
                dummy_depth_features = torch.zeros(obs.shape[0], c_out, h_out, w_out)
            else:
                output_dim = self.estimator.depth_encoder.output_dim
                dummy_depth_features = torch.zeros(obs.shape[0], output_dim)
            rnn_layers = self.estimator.rnn.rnn.num_layers
            hidden_size = self.estimator.rnn.rnn.hidden_size
            dummy_hidden_state = torch.zeros(rnn_layers, obs.shape[0], hidden_size)
            obs, obs_hist, dummy_depth_features, dummy_hidden_state = obs.to("cpu"), obs_hist.to("cpu"), dummy_depth_features.to("cpu"), dummy_hidden_state.to("cpu")
            torch.onnx.export(
                policy,
                (obs, obs_hist, dummy_depth_features, dummy_hidden_state),
                export_path,
                input_names=["obs", "obs_hist", "depth_features", "hidden_state"],
                output_names=["actions", "next_hidden_state"],
                opset_version=17,
                optimize=False,  # Optimize later
                export_params=True,
                dynamic_axes={
                    "obs": {0: "batch_size"},
                    "obs_hist": {0: "batch_size"},
                    "depth_features": {0: "batch_size"},
                    "hidden_state": {1: "batch_size"},
                    "actions": {0: "batch_size"},
                    "next_hidden_state": {1: "batch_size"},
                },
                do_constant_folding=True,
            )

            import onnx

            onnx.checker.check_model(export_path, full_check=True)
            onnx_model = onnx.load(export_path)
            optimize_onnx_model(onnx_model, export_path, verbose=True)
            onnx.checker.check_model(export_path, full_check=True)


class Estimator(nn.Module):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        proprio_encoder_hidden_dims: list[int],
        proprio_encoder_output_dim: int,
        depth_encoder_channels: list[int],
        depth_encoder_output_channel: int,
        depth_encoder_kernel_sizes: list[int],
        depth_encoder_strides: list[int],
        depth_encoder_padding: str,
        depth_encoder_norm: list[str] | str,
        depth_encoder_flatten: bool,
        transformer_encoder_embedding_dim: int,
        transformer_encoder_num_heads: int,
        transformer_encoder_num_layers: int,
        transformer_encoder_mlp_ratio: float,
        transformer_encoder_dropout: float,
        rnn_type: str,
        rnn_hidden_size: int,
        rnn_num_layers: int,
        prediction_heads_hidden_dims: dict[str, list[int] | None],
        decoder_hidden_dims: list[int],
        height_scan_decoder_hidden_dims: list[int],
        estimated_state_dims: dict[str, int],
        height_scan_estimation_vector_dim: int,
        latent_vector_dim: int,
        height_scan_decoder_output_dim: int,
        activation: str,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "Estimator.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )
        super().__init__()

        # Get all the input and output dimensions
        proprio_encoder_input_dim = 0
        decoder_output_dim = 0
        for obs_group in obs_groups["policy"]:
            if obs_group == "exteroception":
                continue
            proprio_encoder_input_dim += (obs[obs_group].shape[-1] * obs[obs_group].shape[1])
            decoder_output_dim += obs[obs_group].shape[-1]

        depth_encoder_input_dim = (obs["exteroception"].shape[-3], obs["exteroception"].shape[-2])
        depth_encoder_input_channel = obs["exteroception"].shape[1]
        depth_encoder_output_channels = depth_encoder_channels + [depth_encoder_output_channel]

        decoder_input_dim = height_scan_estimation_vector_dim + latent_vector_dim
        for key in estimated_state_dims:
            decoder_input_dim += estimated_state_dims[key]

        # Proprioception encoder
        self.proprio_encoder = MLP(proprio_encoder_input_dim, proprio_encoder_output_dim, proprio_encoder_hidden_dims, activation)

        # Depth encoder
        self.depth_encoder = CNN(
            input_dim=depth_encoder_input_dim,
            input_channels=depth_encoder_input_channel,
            output_channels=depth_encoder_output_channels,
            kernel_size=depth_encoder_kernel_sizes,
            stride=depth_encoder_strides,
            padding=depth_encoder_padding,
            norm=depth_encoder_norm,
            activation=activation,
            flatten=depth_encoder_flatten,
        )

        # Transformer encoder
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_encoder_embedding_dim,
            nhead=transformer_encoder_num_heads,
            dim_feedforward=int(transformer_encoder_embedding_dim * transformer_encoder_mlp_ratio),
            dropout=transformer_encoder_dropout,
            activation="gelu",
            batch_first=False,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_encoder_layer,
            num_layers=transformer_encoder_num_layers,
            norm=nn.LayerNorm(transformer_encoder_embedding_dim),
        )
        H_out, W_out = self.depth_encoder.output_dim
        num_tokens = 1 + H_out * W_out
        self.pos_embed = nn.Parameter(torch.zeros(num_tokens, 1, transformer_encoder_embedding_dim))

        # RNN (GRU/LSTM)
        self.rnn = Memory(transformer_encoder_embedding_dim * 2, rnn_hidden_size, rnn_num_layers, rnn_type)

        # Prediction heads
        for key in estimated_state_dims:
            if prediction_heads_hidden_dims[key] is not None:
                setattr(self, f"prediction_head_{key}", MLP(rnn_hidden_size, estimated_state_dims[key], prediction_heads_hidden_dims[key], activation))
            else:
                setattr(self, f"prediction_head_{key}", nn.Linear(rnn_hidden_size, estimated_state_dims[key]))
        if prediction_heads_hidden_dims["height_scan_estimation_vector"] is not None:
            self.prediction_head_height_scan_estimation_vector = MLP(rnn_hidden_size, height_scan_estimation_vector_dim, prediction_heads_hidden_dims["height_scan_estimation_vector"], activation)
        else:
            self.prediction_head_height_scan_estimation_vector = nn.Linear(rnn_hidden_size, height_scan_estimation_vector_dim)
        if prediction_heads_hidden_dims["latent_vector"] is not None:
            self.prediction_head_latent_vector = MLP(rnn_hidden_size, [2, latent_vector_dim], prediction_heads_hidden_dims["latent_vector"], activation)
        else:
            layers = []
            layers.append(nn.Linear(rnn_hidden_size, latent_vector_dim * 2))
            layers.append(nn.Unflatten(dim=-1, unflattened_size=[2, latent_vector_dim]))
            self.prediction_head_latent_vector = nn.Sequential(*layers)
        self.z_m_norm = nn.LayerNorm(height_scan_estimation_vector_dim)

        # Decoder
        self.decoder = MLP(decoder_input_dim, decoder_output_dim, decoder_hidden_dims, activation)

        # Height scan decoder
        self.height_scan_decoder = MLP(height_scan_estimation_vector_dim, height_scan_decoder_output_dim, height_scan_decoder_hidden_dims, activation)

    def encode(self, obs_hist: torch.Tensor, depth_images: torch.Tensor, masks: torch.Tensor | None = None, hidden_state: HiddenState = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lead_shape = obs_hist.shape[:-1]

        # flatten the leading dimensions
        obs_hist_flat = obs_hist.reshape(-1, obs_hist.shape[-1])
        depth_images_flat = depth_images.reshape(-1, *depth_images.shape[-3:])

        # transformer fusion
        fused_feature_flat = self._encode_step(obs_hist_flat, depth_images_flat)
        fused_feature = fused_feature_flat.reshape(*lead_shape, -1)

        # RNN encoding
        memory_out = self.rnn(fused_feature, masks, hidden_state).squeeze(0)

        # explicit heads
        encode_lin_vel = self.prediction_head_lin_vel(memory_out)
        encode_feet_clearances = self.prediction_head_feet_clearances(memory_out)

        # implicit heads
        z_m = self.prediction_head_height_scan_estimation_vector(memory_out)
        z_m = self.z_m_norm(z_m)
        latent_mean_and_logvar = self.prediction_head_latent_vector(memory_out)
        z_mean, z_logvar = torch.unbind(latent_mean_and_logvar, dim=-2)
        z_logvar = (0.5 * z_logvar).exp().clip(1.0e-6, 5.0).square().log()
        z = self._reparameterize(z_mean, z_logvar)

        return encode_lin_vel, encode_feet_clearances, z_m, z, z_mean, z_logvar

    def decode(self, estimated_states: torch.Tensor, height_scan_estimation_vector: torch.Tensor, latent_vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reconstructed_obs = self.decoder(torch.cat((estimated_states.detach(), latent_vector), dim=-1))
        reconstructed_height_scan = self.height_scan_decoder(height_scan_estimation_vector)
        return reconstructed_obs, reconstructed_height_scan

    def _encode_step(self, obs_hist_flat: torch.Tensor, depth_images_flat: torch.Tensor) -> torch.Tensor:
        # 1) unimodal encoding
        proprio_feature = self.proprio_encoder(obs_hist_flat)
        depth_features = self.depth_encoder(depth_images_flat)

        # 2) tokenization and positional encoding
        proprio_token = proprio_feature.unsqueeze(0)
        depth_tokens = depth_features.flatten(-2).permute(2, 0, 1)
        mixed_tokens = torch.cat((proprio_token, depth_tokens), dim=0) + self.pos_embed

        # 3) transformer fusion
        mixed_tokens = self.transformer_encoder(mixed_tokens)

        # 4）token aggregation
        proprio_token = mixed_tokens[0]
        depth_tokens_mean = mixed_tokens[1:].mean(dim=0)
        fused_feature = torch.cat((proprio_token, depth_tokens_mean), dim=-1)

        return fused_feature

    def _reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std


class InferenceWrapper(nn.Module):
    def __init__(self, models: ActorCriticPIE) -> None:
        super().__init__()

        self.proprio_encoder = models.estimator.proprio_encoder
        self.pos_embed = models.estimator.pos_embed
        self.transformer_encoder = models.estimator.transformer_encoder
        self.rnn = models.estimator.rnn.rnn
        self.prediction_head_lin_vel = models.estimator.prediction_head_lin_vel
        self.prediction_head_feet_clearances = models.estimator.prediction_head_feet_clearances
        self.prediction_head_height_scan_estimation_vector = models.estimator.prediction_head_height_scan_estimation_vector
        self.prediction_head_latent_vector = models.estimator.prediction_head_latent_vector
        self.actor = models.actor

        self.actor_obs_normalizer = models.actor_obs_normalizer
        self.actor_lin_vel_normalizer = models.actor_lin_vel_normalizer
        self.actor_feet_clearances_normalizer = models.actor_feet_clearances_normalizer
        self.z_m_norm = models.estimator.z_m_norm

        self.state_dependent_std = models.state_dependent_std

        self.eval()

    def forward(self, obs: torch.Tensor, obs_hist: torch.Tensor, depth_features: torch.Tensor, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 1) normalization
        obs_normalized = self.actor_obs_normalizer(obs)
        obs_hist_normalized = self.actor_obs_normalizer(obs_hist.reshape(-1, obs.shape[-1])).reshape(obs.shape[0], -1)

        # 2) unimodal encoding
        proprio_feature = self.proprio_encoder(obs_hist_normalized)

        # 3) tokenization and positional encoding
        proprio_token = proprio_feature.unsqueeze(0)
        depth_tokens = depth_features.flatten(-2).permute(2, 0, 1)
        mixed_tokens = torch.cat((proprio_token, depth_tokens), dim=0) + self.pos_embed

        # 4) transformer fusion
        mixed_tokens = self.transformer_encoder(mixed_tokens)

        # 5) token aggregation
        proprio_token = mixed_tokens[0]
        depth_tokens_mean = mixed_tokens[1:].mean(dim=0)
        fused_feature = torch.cat((proprio_token, depth_tokens_mean), dim=-1)

        # 6) RNN encoding
        memory_out, next_hidden_state = self.rnn(fused_feature.unsqueeze(0), hidden_state)
        memory_out = memory_out.squeeze(0)

        # 7) explicit heads
        encode_lin_vel = self.prediction_head_lin_vel(memory_out)
        encode_feet_clearances = self.prediction_head_feet_clearances(memory_out)

        # 8) implicit heads
        z_m = self.prediction_head_height_scan_estimation_vector(memory_out)
        z_m = self.z_m_norm(z_m)
        latent_mean_and_logvar = self.prediction_head_latent_vector(memory_out)
        z_mean, _ = torch.unbind(latent_mean_and_logvar, dim=-2)

        # 9) normalization
        lin_vel_normalized = self.actor_lin_vel_normalizer(encode_lin_vel)
        feet_clearances_normalized = self.actor_feet_clearances_normalizer(encode_feet_clearances)

        # 10) inference
        actor_input = torch.cat((obs_normalized, lin_vel_normalized, feet_clearances_normalized, z_m, z_mean), dim=-1)
        if self.state_dependent_std:
            return self.actor(actor_input)[..., 0, :], next_hidden_state
        else:
            return self.actor(actor_input), next_hidden_state
