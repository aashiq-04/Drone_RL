MAX_EPISODES = 300
MAX_STEPS = 200
BATCH_SIZE = 128
BUFFER_SIZE = int(1e7)
STATE_DIM = 18  # (position, velocity, LIDAR, IMU)
ACTION_DIM = 3  # (vx, vy, vz)
MAX_ACTION = 1.0
NUM_EPISODES = 5  # Number of test episodes