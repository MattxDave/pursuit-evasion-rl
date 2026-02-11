"""Quick test of the updated environment."""
from src.multi_environment import MultiAgentPursuitEnv, EvaderTrainingEnv
from src.multi_config import get_default_multi_config
import numpy as np

config = get_default_multi_config()
env = MultiAgentPursuitEnv(config, train_pursuers=True, train_evaders=True)

print('=== Environment Specs ===')
print(f'Pursuer obs shape: {env.pursuer_observation_space.shape}')
print(f'Evader obs shape: {env.evader_observation_space.shape}')
print(f'Pursuer action shape: {env.pursuer_action_space.shape}')
print(f'Evader action shape: {env.evader_action_space.shape}')

obs, info = env.reset()
print(f"\nPursuer obs actual: {obs['pursuers'].shape}")
print(f"Evader obs actual: {obs['evaders'].shape}")

# Test step
action = {
    'pursuers': np.random.uniform(0, 1, (env.num_pursuers, 2)).astype(np.float32),
    'evaders': np.random.uniform(0, 1, (env.num_evaders, 3)).astype(np.float32),
}
obs, rewards, term, trunc, info = env.step(action)
print(f"\nStep successful!")
print(f"Pursuer rewards: {rewards['pursuers']}")
print(f"Evader rewards: {rewards['evaders']}")
print(f"Evader roles: {info['evader_roles']}")

# Run a few more steps to test termination
for i in range(50):
    action = {
        'pursuers': np.random.uniform(0, 1, (env.num_pursuers, 2)).astype(np.float32),
        'evaders': np.random.uniform(0, 1, (env.num_evaders, 3)).astype(np.float32),
    }
    obs, rewards, term, trunc, info = env.step(action)
    if term or trunc:
        print(f"\nEpisode ended at step {info['step']}")
        print(f"Terminated: {term}, Truncated: {trunc}")
        break

env.close()
print('\n=== All checks passed! ===')
