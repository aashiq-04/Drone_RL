import airsim
import numpy as np

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()

# Enable API control and arm the drone
client.enableApiControl(True, "Drone1")
client.armDisarm(True, "Drone1")

# Take off
client.takeoffAsync(vehicle_name="Drone1").join()

# Fetch and print sensor data
try:
    # LiDAR data
    front_lidar = client.getLidarData("FrontLidar", "Drone1")
    left_lidar = client.getLidarData("LeftLidar", "Drone1")
    right_lidar = client.getLidarData("RightLidar", "Drone1")
    bottom_lidar = client.getLidarData("BottomLidar", "Drone1")

    print("Front LiDAR Points:", len(front_lidar.point_cloud))
    print("Left LiDAR Points:", len(left_lidar.point_cloud))
    print("Right LiDAR Points:", len(right_lidar.point_cloud))
    print("Bottom LiDAR Points:", len(bottom_lidar.point_cloud))

    # GPS data
    gps_data = client.getGpsData(vehicle_name="Drone1")
    print("GPS Data:", gps_data.gnss.geo_point)

    # IMU data
    imu_data = client.getImuData(vehicle_name="Drone1")
    print("IMU Orientation:", imu_data.orientation)
    print("IMU Linear Acceleration:", imu_data.linear_acceleration)
except Exception as e:
    print("Error reading sensor data:", e)

# Land and reset
client.landAsync(vehicle_name="Drone1").join()
client.armDisarm(False, "Drone1")
client.enableApiControl(False, "Drone1")
    