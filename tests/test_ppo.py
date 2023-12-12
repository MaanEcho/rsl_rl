import torch
import unittest
from rsl_rl.algorithms import PPO
from rsl_rl.env.pole_balancing import PoleBalancing


class FakeCritic(torch.nn.Module):
    def __init__(self, values):
        self.recurrent = False
        self.values = values

    def forward(self, _):
        return self.values


class PPOTest(unittest.TestCase):
    def test_gae_computation(self):
        # GT taken from old PPO implementation.

        env = PoleBalancing(environment_count=4)
        ppo = PPO(env, device="cpu", gae_lambda=0.97, gamma=0.99)

        rewards = torch.tensor(
            [
                [-1.0000e02, -1.4055e-01, -3.0476e-02, -2.7149e-01, -1.1157e-01, -2.3366e-01, -3.3658e-01, -1.6447e-01],
                [
                    -1.7633e-01,
                    -2.6533e-01,
                    -3.0786e-01,
                    -2.6038e-01,
                    -2.7176e-01,
                    -2.1655e-01,
                    -1.5441e-01,
                    -2.9580e-01,
                ],
                [-1.5952e-01, -1.5177e-01, -1.4296e-01, -1.6131e-01, -3.1395e-02, 2.8808e-03, -3.1242e-02, 4.8696e-03],
                [1.1407e-02, -1.0000e02, -6.2290e-02, -3.7030e-01, -2.7648e-01, -3.6655e-01, -2.8456e-01, -2.3165e-01],
            ]
        )
        dones = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
            ]
        )
        observations = torch.zeros((dones.shape[0], dones.shape[1], 24))
        timeouts = torch.zeros((dones.shape[0], dones.shape[1]))
        values = torch.tensor(
            [
                [-4.6342, -7.6510, -7.0166, -7.6137, -7.4130, -7.7071, -7.7413, -7.8301],
                [-7.0442, -7.0032, -6.9321, -6.7765, -6.5433, -6.3503, -6.2529, -5.9337],
                [-7.5753, -7.8146, -7.6142, -7.8443, -7.8791, -7.7973, -7.7853, -7.7724],
                [-6.4326, -6.1673, -7.6511, -7.7505, -8.0004, -7.8584, -7.5949, -7.9023],
            ]
        )
        last_values = torch.tensor([-7.9343, -5.8734, -7.8527, -8.1257])

        ppo.critic = FakeCritic(last_values)
        dataset = [
            {
                "dones": dones[:, i],
                "next_critic_observations": observations[:, i],
                "rewards": rewards[:, i],
                "timeouts": timeouts[:, i],
                "values": values[:, i],
            }
            for i in range(dones.shape[1])
        ]

        processed_dataset = ppo._process_dataset(dataset)
        processed_returns = torch.stack(
            [processed_dataset[i]["advantages"] + processed_dataset[i]["values"] for i in range(dones.shape[1])],
            dim=-1,
        )
        processed_advantages = torch.stack(
            [processed_dataset[i]["normalized_advantages"] for i in range(dones.shape[1])], dim=-1
        )

        expected_returns = torch.tensor(
            [
                [-100.0000, -8.4983, -8.4863, -8.5699, -8.4122, -8.4054, -8.2702, -8.0194],
                [-7.2900, -7.1912, -6.9978, -6.7569, -6.5627, -6.3547, -6.1985, -6.1104],
                [-7.9179, -7.8374, -7.7679, -7.6976, -7.6041, -7.6446, -7.7229, -7.7693],
                [-96.2018, -100.0000, -9.0710, -9.1415, -8.8863, -8.7228, -8.4668, -8.2761],
            ]
        )
        expected_advantages = torch.tensor(
            [
                [-3.1452, 0.3006, 0.2779, 0.2966, 0.2951, 0.3060, 0.3122, 0.3246],
                [0.3225, 0.3246, 0.3291, 0.3322, 0.3308, 0.3313, 0.3335, 0.3250],
                [0.3190, 0.3307, 0.3259, 0.3368, 0.3415, 0.3371, 0.3338, 0.3316],
                [-2.9412, -3.0893, 0.2797, 0.2808, 0.2992, 0.3000, 0.2997, 0.3179],
            ]
        )

        self.assertTrue(torch.isclose(processed_returns, expected_returns, atol=1e-4).all())
        self.assertTrue(torch.isclose(processed_advantages, expected_advantages, atol=1e-4).all())
