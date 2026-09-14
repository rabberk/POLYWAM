from dataclasses import dataclass, field
from types import SimpleNamespace

import torch
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig

from prismatic.models.flow_matching_head.action_encoder import ActionEncoder
from prismatic.models.flow_matching_head.cross_attention_dit import DiT


def flow_matching_training_target(
    action: torch.Tensor,
    noise: torch.Tensor,
    prediction_type: str,
) -> torch.Tensor:
    if prediction_type == "velocity":
        return action - noise
    if prediction_type == "x":
        return action
    raise ValueError(f"unsupported flow prediction_type: {prediction_type!r}")


def x_prediction_to_velocity(
    predicted_clean: torch.Tensor,
    current: torch.Tensor,
    t: float | torch.Tensor,
) -> torch.Tensor:
    denominator = torch.as_tensor(1.0 - t, dtype=current.dtype, device=current.device).clamp_min(1e-6)
    return (predicted_clean - current) / denominator


def weighted_action_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    step_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute step-weighted action MSE with a scale matching ordinary mean MSE."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share shape [B,T,D]")
    per_step = (prediction - target).square().mean(dim=-1)
    if step_weights is None:
        return per_step.mean()
    weights = torch.as_tensor(step_weights, dtype=per_step.dtype, device=per_step.device)
    if weights.ndim != 1 or weights.numel() != prediction.shape[1] or torch.any(weights <= 0):
        raise ValueError("step_weights must contain one positive value per action step")
    weights = weights / weights.mean()
    return (per_step * weights[None]).mean()


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(torch.relu(self.layer1(x)))


@dataclass
class FlowMatchingActionHeadConfig(PretrainedConfig):
    add_pos_embed: bool = field(default=True)
    diffusion_model_cfg: dict = field(default=None)
    input_embedding_dim: int = field(default=768)
    hidden_size: int = field(default=1024)
    max_seq_len: int = field(default=1024)
    action_dim: int = field(default=None)
    action_horizon: int = field(default=None)
    noise_beta_alpha: float = field(default=1.5)
    noise_beta_beta: float = field(default=1.0)
    noise_s: float = field(default=0.999)
    num_timestep_buckets: int = field(default=1000)
    num_inference_timesteps: int = field(default=4)
    num_target_vision_tokens: int = field(default=32)
    prediction_type: str = field(default="velocity")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FlowMatchingActionHead(nn.Module):
    """
    VLA-JEPA GR00T action head adapted to JEPA-WAM.

    The architecture is copied from VLA-JEPA; the main interface change is that
    it consumes the existing action-placeholder hidden sequence from Prismatic
    instead of custom `<|embodied_action|>` tokenizer tokens.
    """

    def __init__(
        self,
        d_proprio: int,
        d_action: int,
        d_llm: int,
        horizon: int,
        fm_hidden_size: int = 1024,
        fm_num_layers: int = 16,
        fm_num_inference_timesteps: int = 4,
        fm_num_timestep_buckets: int = 1000,
        fm_noise_beta_alpha: float = 1.5,
        fm_noise_beta_beta: float = 1.0,
        fm_noise_s: float = 0.999,
        fm_num_target_vision_tokens: int = 32,
        fm_add_pos_embed: bool = True,
        fm_max_seq_len: int = 1024,
        fm_state_dropout: float = 0.5,
        prediction_type: str = "velocity",
    ):
        super().__init__()

        action_model_cfg = {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32}
        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]
        diffusion_model_cfg = {
            **action_model_cfg,
            "cross_attention_dim": d_llm,
            "dropout": 0.2,
            "final_dropout": True,
            "interleave_self_attention": True,
            "norm_type": "ada_norm",
            "num_layers": fm_num_layers,
            "output_dim": fm_hidden_size,
            "positional_embeddings": None,
        }

        config = FlowMatchingActionHeadConfig(
            add_pos_embed=fm_add_pos_embed,
            diffusion_model_cfg=diffusion_model_cfg,
            input_embedding_dim=self.input_embedding_dim,
            hidden_size=fm_hidden_size,
            max_seq_len=fm_max_seq_len,
            action_dim=d_action,
            action_horizon=horizon,
            noise_beta_alpha=fm_noise_beta_alpha,
            noise_beta_beta=fm_noise_beta_beta,
            noise_s=fm_noise_s,
            num_timestep_buckets=fm_num_timestep_buckets,
            num_inference_timesteps=fm_num_inference_timesteps,
            num_target_vision_tokens=fm_num_target_vision_tokens,
            prediction_type=prediction_type,
        )
        flow_matching_training_target(torch.zeros(1), torch.zeros(1), config.prediction_type)
        self.config = config
        self.full_config = SimpleNamespace(framework=SimpleNamespace(action_model=config))

        self.hidden_size = config.hidden_size
        self.model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = MLP(
            input_dim=d_proprio,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.state_dropout = nn.Dropout(p=fm_state_dropout)
        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        # Per-embodiment adapters. The action dimension only enters at the two ends
        # (encode actions in, decode actions out) plus the proprio encoder, so
        # co-training across robots needs nothing more than a swap of those three;
        # the DiT trunk, future tokens and positional embedding stay shared.
        # Empty by default, so single-robot runs and existing checkpoints are unchanged.
        self.embodiment_adapters = nn.ModuleDict()
        self._embodiment_shapes: dict[str, tuple[int, int]] = {}

        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def _prediction_type(self) -> str:
        """Keep compatibility with lightweight legacy tests that bypass ``__init__``."""
        config = getattr(self, "config", None)
        return "velocity" if config is None else config.prediction_type

    def _prepare_state(self, proprio: torch.Tensor) -> torch.Tensor:
        if proprio.dim() == 3:
            proprio = proprio[:, 0, :]
        return proprio

    def _prediction_type(self) -> str:
        config = getattr(self, "config", None)
        return "velocity" if config is None else config.prediction_type

    def register_embodiment(
        self, name: str, *, d_action: int, d_proprio: int, action_horizon: int | None = None
    ) -> None:
        """Add the three action-space-specific modules for another robot."""
        if name in self.embodiment_adapters:
            raise ValueError(f"embodiment {name!r} is already registered")
        if d_action <= 0 or d_proprio <= 0:
            raise ValueError("action and proprio dimensions must be positive")
        horizon = self.action_horizon if action_horizon is None else int(action_horizon)
        if horizon <= 0:
            raise ValueError("action_horizon must be positive")
        self._embodiment_shapes[name] = (horizon, int(d_action))
        self.embodiment_adapters[name] = nn.ModuleDict(
            {
                "state_encoder": MLP(
                    input_dim=d_proprio, hidden_dim=self.hidden_size, output_dim=self.input_embedding_dim
                ),
                "action_encoder": ActionEncoder(action_dim=d_action, hidden_size=self.input_embedding_dim),
                "action_decoder": MLP(
                    input_dim=self.hidden_size, hidden_dim=self.hidden_size, output_dim=d_action
                ),
            }
        )

    def _adapter(self, embodiment: str | None):
        """The default modules unless this batch names a registered embodiment."""
        if embodiment is None or embodiment not in self.embodiment_adapters:
            return self.state_encoder, self.action_encoder, self.action_decoder
        adapter = self.embodiment_adapters[embodiment]
        return adapter["state_encoder"], adapter["action_encoder"], adapter["action_decoder"]

    def _predict_velocity(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        timesteps_tensor: torch.Tensor,
        state: torch.Tensor,
        embodiment: str | None = None,
    ):
        device = vl_embs.device
        compute_dtype = vl_embs.dtype
        actions = actions.to(dtype=compute_dtype)
        if state is not None:
            state = state.to(dtype=compute_dtype)
        state_encoder, action_encoder, action_decoder = self._adapter(embodiment)
        action_features = action_encoder(actions, timesteps_tensor)
        state_features = state_encoder(state).unsqueeze(1) if state is not None else None
        if state_features is not None:
            state_features = self.state_dropout(state_features)

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = (
            torch.cat((state_features, future_tokens, action_features), dim=1)
            if state_features is not None
            else torch.cat((future_tokens, action_features), dim=1)
        )
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=timesteps_tensor,
            return_all_hidden_states=False,
        )
        pred = action_decoder(model_output)
        return pred[:, -actions.shape[1] :]

    def forward(
        self,
        vl_embs: torch.Tensor,
        proprio: torch.Tensor,
        action_gt: torch.Tensor,
        action_step_weights: torch.Tensor | None = None,
        embodiment: str | None = None,
        **_: object,
    ):
        compute_dtype = vl_embs.dtype
        state = self._prepare_state(proprio).to(dtype=compute_dtype)
        action_gt = action_gt.to(dtype=compute_dtype)
        noise = torch.randn(action_gt.shape, device=action_gt.device, dtype=action_gt.dtype)
        t = self.sample_time(action_gt.shape[0], device=action_gt.device, dtype=action_gt.dtype)
        t = t[:, None, None]

        noisy_trajectory = (1 - t) * noise + t * action_gt
        target = flow_matching_training_target(action_gt, noise, self._prediction_type())

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        extra = {"embodiment": embodiment} if embodiment is not None else {}
        pred_actions = self._predict_velocity(vl_embs, noisy_trajectory, t_discretized, state, **extra)
        loss = weighted_action_mse(pred_actions, target, action_step_weights)
        return loss, pred_actions

    @torch.no_grad()
    def predict_action(self, vl_embs: torch.Tensor, proprio: torch.Tensor, embodiment: str | None = None) -> torch.Tensor:
        state = self._prepare_state(proprio).to(dtype=vl_embs.dtype)
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        horizon, action_dim = self._embodiment_shapes.get(
            embodiment, (self.action_horizon, self.action_dim)
        )
        actions = torch.randn(
            size=(batch_size, horizon, action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device, dtype=torch.long)
            extra = {"embodiment": embodiment} if embodiment is not None else {}
            model_prediction = self._predict_velocity(vl_embs, actions, timesteps_tensor, state, **extra)
            pred_velocity = (
                x_prediction_to_velocity(model_prediction, actions, t_cont)
                if self._prediction_type() == "x"
                else model_prediction
            )
            actions = actions + dt * pred_velocity
        return actions

    @torch.no_grad()
    def sample_action(self, vl_embs: torch.Tensor, proprio: torch.Tensor, num_steps: int | None = None) -> torch.Tensor:
        if num_steps is not None:
            old_steps = self.num_inference_timesteps
            self.num_inference_timesteps = num_steps
            try:
                return self.predict_action(vl_embs, proprio)
            finally:
                self.num_inference_timesteps = old_steps
        return self.predict_action(vl_embs, proprio)
