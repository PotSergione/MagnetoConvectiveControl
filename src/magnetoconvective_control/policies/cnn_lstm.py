import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ------------------------------------------------------------------------
# Custom feature extractor for channel-first wall observations.
# ------------------------------------------------------------------------

class CNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 128):
        heat_flux_space = observation_space.spaces["heat_flux"]
        action_space = observation_space.spaces["previous_actions"]
        if len(heat_flux_space.shape) != 3:
            raise ValueError("heat-flux observations must be channel-first images")
        if action_space.shape != (10,):
            raise ValueError("previous_actions observations must have shape (10,)")

        super().__init__(observation_space, features_dim + action_space.shape[0])
        self.cnn = th.nn.Sequential(
            th.nn.ConstantPad2d((0, 0, 1, 1), 0.0),
            th.nn.CircularPad2d((1, 1, 0, 0)),
            th.nn.Conv2d(heat_flux_space.shape[0], 32, kernel_size=3, stride=2, padding=0),
            th.nn.LeakyReLU(),
            th.nn.ConstantPad2d((0, 0, 1, 1), 0.0),
            th.nn.CircularPad2d((1, 1, 0, 0)),
            th.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),  # 64->32
            th.nn.LeakyReLU(),
            th.nn.ConstantPad2d((0, 0, 1, 1), 0.0),
            th.nn.CircularPad2d((1, 1, 0, 0)),
            th.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),  # 32->16
            th.nn.LeakyReLU(),
            th.nn.Flatten(),
        )

        # Compute the output size of the CNN to define the linear layer
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(heat_flux_space.sample()[None]).float()
            ).shape[1]

        self.linear = th.nn.Sequential(th.nn.Linear(n_flatten, features_dim), th.nn.LeakyReLU())

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        heat_flux_features = self.linear(self.cnn(observations["heat_flux"]))
        return th.cat((heat_flux_features, observations["previous_actions"]), dim=1)
