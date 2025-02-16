import numpy as np
import matplotlib.pyplot as plt
from src_sac.Environment import DroneEnv3D

# Create the environment
env = DroneEnv3D()

# Reset the environment to its initial state
obs = env.reset()

# Render the initial state
env.render()

# Keep the plot open
plt.show()
