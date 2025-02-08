from stable_baselines3 import PPO
from sample import DroneEnv3D
import time  # Import the time module

# Load the trained model
# model = PPO.load("ppo_drone_navigation_tuned")
try:
    model = PPO.load("ppo_drone_navigation_tuned.zip")
except Exception as e:
    print(f"Error loading model: {e}")

# Create the environment
env = DroneEnv3D()
obs = env.reset()
done = False

# Test the trained model
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    print(f"Position: {obs}, Reward: {reward}, Done: {done}")
    
    # Add a delay to slow down the visualization
    time.sleep(0.2)  # Adjust the delay (in seconds) as needed
time.sleep(5)