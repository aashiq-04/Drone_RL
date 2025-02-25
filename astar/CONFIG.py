MAX_EPISODES = 500
MAX_STEPS = 500
BATCH_SIZE = 128
BUFFER_SIZE = int(1e6)
STATE_DIM = 21  # (position, velocity, LIDAR, IMU)
ACTION_DIM = 3  # (vx, vy, vz)
MAX_ACTION = 1.0
NUM_EPISODES = 5 # Number of test episodes