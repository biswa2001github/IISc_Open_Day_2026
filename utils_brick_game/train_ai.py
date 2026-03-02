from stable_baselines3 import PPO
from brick_env import BrickBreakerEnv

env = BrickBreakerEnv()
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
)

model.learn(total_timesteps=1_500_000)
model.save("brick_ai")