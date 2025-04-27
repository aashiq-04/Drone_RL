import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from SAC_Model.SAC import SACAgent
from PPO.ENV import DroneEnv3D
from SAC_Model.Env import DroneEnv3D as DroneENVSAC 
from SAC_Model.CONFIG import STATE_DIM, ACTION_DIM, MAX_ACTION, NUM_EPISODES, MAX_STEPS, MAP_SIZE
import os
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv


# Import stable-baselines3 models
try:
    from stable_baselines3 import PPO, DDPG, TD3
    SB3_AVAILABLE = True
except ImportError:
    print("Warning: stable-baselines3 not found, only custom SAC will be evaluated")
    SB3_AVAILABLE = False

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

# Model paths
MODEL_PATHS = {
    "SAC": {
        "actor": "SAC_Model/checkpoints/sac_actor_best.pth",
        "critic": "SAC_Model/checkpoints/sac_critic_best.pth"
    },
    "PPO": "PPO/ppo_drone_updated.zip",
    "DDPG": "DDPG/checkpointDDPG/ddpg_final.zip",
    # "TD3": "checkpoints/td3_model.zip"
}

def evaluate_model(model_name, agent=None, num_episodes=NUM_EPISODES, visualize=False):
    """
    Evaluate a model and return performance metrics
    """
    # Initialize environment
    sac_env = DroneENVSAC()
    env = DroneEnv3D()
    
    # Device configuration
    device = "cpu"
    
    # Load appropriate model
    sb3_model = None
    
    try:
        if model_name == "SAC":
            # Custom SAC model
            agent.actor.load_state_dict(torch.load(MODEL_PATHS[model_name]["actor"], map_location=device))
            agent.critic.load_state_dict(torch.load(MODEL_PATHS[model_name]["critic"], map_location=device))
            agent.actor.to(device)
            agent.critic.to(device)
            print(f"✅ Custom {model_name} model loaded successfully!")
        elif SB3_AVAILABLE:
            # Stable Baselines 3 models
            if model_name == "PPO":
                sb3_model = PPO.load(MODEL_PATHS[model_name])
            elif model_name == "DDPG":
                sb3_model = DDPG.load(MODEL_PATHS[model_name])
            # elif model_name == "TD3":
            #     sb3_model = TD3.load(MODEL_PATHS[model_name])
            print(f"✅ Stable Baselines3 {model_name} model loaded successfully!")
        else:
            if model_name != "SAC":
                print(f"❌ Stable Baselines3 not available, skipping {model_name}")
                return None
        
    except FileNotFoundError:
        print(f"❌ {model_name} model files not found. Please check path: {MODEL_PATHS[model_name]}")
        return None
    except Exception as e:
        print(f"❌ Error loading {model_name} model: {str(e)}")
        return None

    # Setup visualization if needed
    if visualize:
        plt.ion()
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
    
    # Tracking values
    test_rewards = []
    test_collisions = []
    test_steps = []
    test_distances = []
    successful_episodes = 0
    episode_exploration_efficiencies = []
    episode_time_efficiencies = []
    episode_collision_rates = []
    
    # Initialize tracking grids for exploration
    resolution = 0.5  # Each cell represents 0.5x0.5x0.5 cube
    grid_size = int(MAP_SIZE / resolution)
    visited_grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)

    # Calculate total explorable cells (excluding obstacles)
    total_cells = grid_size**3
    # Subtract cells occupied by obstacles (approximate)
    for obs in env.obstacles:
        # Assuming obs contains [x, y, z, radius]
        obs_center = obs[:3]
        obs_radius = obs[3] if len(obs) > 3 else 0.5  # Default radius if not specified
        
        # Create a mask of cells inside this obstacle (approximate)
        for x in range(grid_size):
            for y in range(grid_size):
                for z in range(grid_size):
                    # Convert grid coordinates to world coordinates
                    world_x = x * resolution
                    world_y = y * resolution
                    world_z = z * resolution
                    
                    # Check if point is inside obstacle
                    dist = np.sqrt((world_x - obs_center[0])**2 + 
                                  (world_y - obs_center[1])**2 + 
                                  (world_z - obs_center[2])**2)
                    if dist <= obs_radius:
                        visited_grid[x, y, z] = True  # Mark as non-explorable
        
    # Calculate explorable cells
    explorable_cells = np.sum(~visited_grid)
    # Reset visited grid for tracking
    visited_grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    
    # Start episodes
    start_time = time.time()
    
    for episode in range(num_episodes):
        state = env.reset()  
        state_sac = sac_env.reset()
        total_reward = 0
        episode_visited_cells = set()  # Track unique positions visited in this episode
        collision_count = 0
        total_distance = 0
        last_position = env.drone_pos.copy()
        
        env.path = [env.drone_pos.copy()]  # Store drone path
        path_array = np.array(env.path)  # Precompute path array

        for step in range(MAX_STEPS):
            # Get action based on model type
            if model_name == "SAC":
                # Custom SAC model
                with torch.no_grad():
                    state_tensor = torch.from_numpy(state_sac).float().unsqueeze(0).to(device)
                    action = agent.select_action(state_tensor).flatten()
            else:
                # Stable Baselines3 models
                action, _ = sb3_model.predict(state, deterministic=True)

            next_state, reward, done, truncated,_ = env.step(action)
            total_reward += reward
            
            # Calculate distance traveled in this step
            current_position = env.drone_pos.copy()
            step_distance = np.linalg.norm(np.array(current_position) - np.array(last_position))
            total_distance += step_distance
            last_position = current_position
            
            # Track visited cells for exploration efficiency
            grid_x = min(max(int(env.drone_pos[0] / resolution), 0), grid_size-1)
            grid_y = min(max(int(env.drone_pos[1] / resolution), 0), grid_size-1)
            grid_z = min(max(int(env.drone_pos[2] / resolution), 0), grid_size-1)
            
            # Mark as visited
            visited_grid[grid_x, grid_y, grid_z] = True
            
            # Track unique position for this episode
            pos_tuple = tuple(np.round(env.drone_pos, 1))  # Round to reduce unique positions
            episode_visited_cells.add(pos_tuple)
            
            state = next_state
            
            # Count collisions
            if env.check_collision():
                collision_count += 1
                
            # Update path efficiently
            env.path.append(env.drone_pos.copy())
            path_array = np.array(env.path)  # Update path once

            # Render if visualization is enabled
            if visualize:
                ax.cla()  # Clear only axes, not the entire figure
                ax.set_xlim(0, env.space_size)
                ax.set_ylim(0, env.space_size)
                ax.set_zlim(0, env.space_size)

                # Plot drone and target
                ax.scatter(*env.drone_pos, color='red', s=100, label="Drone")
                ax.scatter(*env.target, color='green', s=150, label="Target")

                # Plot obstacles
                for obs in env.obstacles:
                    env.draw_cylinder(ax, *obs)

                # Plot path efficiently
                ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2],
                        color='blue', linestyle='-', marker='o', markersize=3, label="Path")

                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_title(f"{model_name} - Episode {episode + 1}")
                ax.legend()

                # Flush events instead of using plt.pause (prevents freezing)
                fig.canvas.flush_events()
                time.sleep(0.01)  # Reduced delay for faster evaluation

            if done:
                if collision_count == 0 and reward > 0:  # Define success criteria
                    successful_episodes += 1
                break
        
        # Store metrics for this episode
        test_rewards.append(total_reward)
        test_collisions.append(collision_count)
        test_steps.append(step + 1)  # +1 because step starts at 0
        test_distances.append(total_distance)
        
        # Calculate metrics for this episode
        episode_visited_count = np.sum(visited_grid)
        exploration_efficiency = (episode_visited_count / explorable_cells) * 100
        time_efficiency = ((MAX_STEPS - (step + 1)) / MAX_STEPS) * 100 if step < MAX_STEPS else 0
        collision_rate = (collision_count / (step + 1)) * 100 if step > 0 else 0
        
        episode_exploration_efficiencies.append(exploration_efficiency)
        episode_time_efficiencies.append(time_efficiency)
        episode_collision_rates.append(collision_rate)
        
        # Print progress
        if episode % 5 == 0 or episode == num_episodes - 1:
            print(f"{model_name} - Episode {episode + 1}/{num_episodes} complete")

    # Calculate overall metrics
    success_rate = (successful_episodes / num_episodes) * 100
    avg_reward = np.mean(test_rewards)
    avg_steps = np.mean(test_steps)
    avg_distance = np.mean(test_distances)
    avg_exploration = np.mean(episode_exploration_efficiencies)
    avg_time_efficiency = np.mean(episode_time_efficiencies)
    avg_collision_rate = np.mean(episode_collision_rates)
    path_efficiency = avg_distance / avg_steps if avg_steps > 0 else 0
    
    # Calculate execution time
    execution_time = time.time() - start_time
    avg_episode_time = execution_time / num_episodes
    
    # Print summary
    print(f"\n📊 {model_name} PERFORMANCE SUMMARY 📊")
    print(f"✅ Success Rate: {success_rate:.2f}%")
    print(f"✅ Average Reward: {avg_reward:.2f}")
    print(f"✅ Average Steps: {avg_steps:.2f}")
    print(f"✅ Average Exploration Efficiency: {avg_exploration:.2f}%")
    print(f"✅ Average Time Efficiency: {avg_time_efficiency:.2f}%")
    print(f"✅ Average Collision Rate: {avg_collision_rate:.2f}%")
    print(f"✅ Path Efficiency: {path_efficiency:.4f}")
    print(f"✅ Average Episode Time: {avg_episode_time:.4f} seconds")
    
    # Clean up
    if visualize:
        plt.ioff()
    env.close()
    
    # Return all metrics in a dictionary
    return {
        "model_name": model_name,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "avg_distance": avg_distance,
        "avg_exploration": avg_exploration,
        "avg_time_efficiency": avg_time_efficiency,
        "avg_collision_rate": avg_collision_rate,
        "path_efficiency": path_efficiency,
        "avg_episode_time": avg_episode_time,
        "rewards": test_rewards,
        "steps": test_steps,
        "collisions": test_collisions,
        "distances": test_distances,
        "exploration_efficiencies": episode_exploration_efficiencies,
        "time_efficiencies": episode_time_efficiencies,
        "collision_rates": episode_collision_rates
    }

def visualize_comparison(results):
    """
    Create comparison visualizations between all models
    """
    if not results:
        print("No results to visualize!")
        return
    
    # Extract model names
    model_names = [result["model_name"] for result in results]
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    plt.suptitle("Drone Navigation: Algorithm Comparison", fontsize=16)
    
    # 1. Bar chart for overall metrics
    metrics = ["success_rate", "avg_exploration", "avg_time_efficiency"]
    metric_names = ["Success Rate (%)", "Exploration Efficiency (%)", "Time Efficiency (%)"]
    
    ax1 = fig.add_subplot(231)
    bar_width = 0.25
    x = np.arange(len(model_names))
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        values = [result[metric] for result in results]
        ax1.bar(x + i*bar_width, values, width=bar_width, label=metric_name)
    
    ax1.set_xlabel("Algorithm")
    ax1.set_ylabel("Performance (%)")
    ax1.set_title("Success and Efficiency Metrics")
    ax1.set_xticks(x + bar_width)
    ax1.set_xticklabels(model_names)
    ax1.legend()
    
    # 2. Bar chart for collision rate (lower is better)
    ax2 = fig.add_subplot(232)
    collision_rates = [result["avg_collision_rate"] for result in results]
    colors = ['green' if rate < 5 else 'orange' if rate < 15 else 'red' for rate in collision_rates]
    ax2.bar(model_names, collision_rates, color=colors)
    ax2.set_xlabel("Algorithm")
    ax2.set_ylabel("Collision Rate (%)")
    ax2.set_title("Average Collision Rate (Lower is Better)")
    for i, v in enumerate(collision_rates):
        ax2.text(i, v + 0.5, f"{v:.1f}%", ha='center')
    
    # 3. Boxplot of rewards
    ax3 = fig.add_subplot(233)
    reward_data = [result["rewards"] for result in results]
    ax3.boxplot(reward_data, labels=model_names)
    ax3.set_ylabel("Reward")
    ax3.set_title("Reward Distribution")
    
    # 4. Line plot showing rewards over episodes
    ax4 = fig.add_subplot(234)
    for result in results:
        ax4.plot(range(1, len(result["rewards"])+1), result["rewards"], label=result["model_name"])
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Reward")
    ax4.set_title("Reward per Episode")
    ax4.legend()
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 5. Path efficiency and average steps
    ax5 = fig.add_subplot(235)
    metrics = ["path_efficiency", "avg_steps"]
    metric_names = ["Path Efficiency", "Avg Steps (÷10)"]
    bar_width = 0.35
    x = np.arange(len(model_names))
    
    # Scale down steps to fit on same chart
    path_efficiencies = [result["path_efficiency"] for result in results]
    avg_steps = [result["avg_steps"]/10 for result in results]  # Divide by 10 to scale
    
    ax5.bar(x - bar_width/2, path_efficiencies, width=bar_width, label="Path Efficiency")
    ax5.bar(x + bar_width/2, avg_steps, width=bar_width, label="Avg Steps (÷10)")
    
    ax5.set_xlabel("Algorithm")
    ax5.set_ylabel("Value")
    ax5.set_title("Path Efficiency and Step Count")
    ax5.set_xticks(x)
    ax5.set_xticklabels(model_names)
    ax5.legend()
    
    # 6. Spider/Radar chart for overall performance
    ax6 = fig.add_subplot(236, polar=True)
    
    # Categories for radar chart, normalized to 0-1 where 1 is always better
    categories = ['Success Rate', 'Exploration', 'Time Efficiency', 
                 'Low Collision Rate', 'Path Efficiency', 'Low Step Count']
    
    # Normalize values between 0 and 1, ensure that higher is always better
    max_values = {
        "success_rate": 100,
        "avg_exploration": 100,
        "avg_time_efficiency": 100,
        "avg_collision_rate": max(max(result["avg_collision_rate"] for result in results), 1),  # Max collision rate
        "path_efficiency": max(max(result["path_efficiency"] for result in results), 0.1),  # Max path efficiency
        "avg_steps": max(max(result["avg_steps"] for result in results), 1)  # Max average steps
    }
    
    # For metrics where lower is better, we invert the normalization
    normalized_results = []
    for result in results:
        normalized = [
            result["success_rate"] / max_values["success_rate"],
            result["avg_exploration"] / max_values["avg_exploration"],
            result["avg_time_efficiency"] / max_values["avg_time_efficiency"],
            1 - (result["avg_collision_rate"] / max_values["avg_collision_rate"]),  # Invert (lower is better)
            result["path_efficiency"] / max_values["path_efficiency"],
            1 - (result["avg_steps"] / max_values["avg_steps"])  # Invert (lower is better)
        ]
        normalized_results.append(normalized)
    
    # Plot radar chart
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    for i, result in enumerate(results):
        values = normalized_results[i]
        values += values[:1]  # Close the loop
        ax6.plot(angles, values, linewidth=2, label=result["model_name"])
        ax6.fill(angles, values, alpha=0.1)
    
    ax6.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax6.set_ylim(0, 1)
    ax6.set_title("Overall Performance Comparison")
    ax6.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("algorithm_comparison.png", dpi=300, bbox_inches='tight')
    print("✅ Comparison visualization saved as 'algorithm_comparison.png'")
    plt.show()
    
    # Create additional detailed comparisons
    create_detailed_comparisons(results)

def create_detailed_comparisons(results):
    """
    Create more detailed comparison visualizations
    """
    # Extract model names
    model_names = [result["model_name"] for result in results]
    
    # 1. Episode-wise exploration efficiency
    plt.figure(figsize=(12, 8))
    for result in results:
        plt.plot(range(1, len(result["exploration_efficiencies"])+1), 
                 result["exploration_efficiencies"], 
                 label=result["model_name"])
    plt.xlabel("Episode")
    plt.ylabel("Exploration Efficiency (%)")
    plt.title("Exploration Efficiency per Episode")
    plt.legend()
    plt.grid(True)
    plt.savefig("exploration_comparison.png", dpi=300)
    print("✅ Exploration comparison saved as 'exploration_comparison.png'")
    
    # 2. Episode-wise collision rates
    plt.figure(figsize=(12, 8))
    for result in results:
        plt.plot(range(1, len(result["collision_rates"])+1), 
                 result["collision_rates"], 
                 label=result["model_name"])
    plt.xlabel("Episode")
    plt.ylabel("Collision Rate (%)")
    plt.title("Collision Rate per Episode")
    plt.legend()
    plt.grid(True)
    plt.savefig("collision_comparison.png", dpi=300)
    print("✅ Collision comparison saved as 'collision_comparison.png'")
    
    # 3. Success vs Steps scatter plot
    plt.figure(figsize=(12, 8))
    for result in results:
        # Define success as episodes with no collisions
        successes = [1 if c == 0 else 0 for c in result["collisions"]]
        steps = result["steps"]
        
        # Calculate marker size based on rewards
        rewards = result["rewards"]
        marker_sizes = [abs(r) * 10 for r in rewards]  # Scale rewards for marker size
        
        plt.scatter(steps, successes, s=marker_sizes, alpha=0.6, label=result["model_name"])
    
    plt.xlabel("Steps Taken")
    plt.ylabel("Success (1=Yes, 0=No)")
    plt.title("Success vs Steps with Reward as Marker Size")
    plt.legend()
    plt.grid(True)
    plt.savefig("success_steps_comparison.png", dpi=300)
    print("✅ Success vs Steps comparison saved as 'success_steps_comparison.png'")
    
    # 4. Time efficiency comparison
    plt.figure(figsize=(12, 8))
    for result in results:
        plt.plot(range(1, len(result["time_efficiencies"])+1), 
                 result["time_efficiencies"], 
                 label=result["model_name"])
    plt.xlabel("Episode")
    plt.ylabel("Time Efficiency (%)")
    plt.title("Time Efficiency per Episode")
    plt.legend()
    plt.grid(True)
    plt.savefig("time_efficiency_comparison.png", dpi=300)
    print("✅ Time efficiency comparison saved as 'time_efficiency_comparison.png'")
    
    # 5. Combined Metric (custom score combining all metrics)
    plt.figure(figsize=(12, 6))
    
    # Calculate combined score (adjust weights as needed)
    combined_scores = []
    for result in results:
        # Higher is better for all except collision rate
        score = (
            0.25 * result["success_rate"] + 
            0.20 * result["avg_exploration"] + 
            0.20 * result["avg_time_efficiency"] + 
            0.20 * (100 - result["avg_collision_rate"]) +  # Invert so higher is better
            0.15 * (result["path_efficiency"] * 100)  # Scale up path efficiency
        ) / 100  # Normalize to 0-1
        
        combined_scores.append(score)
    
    # Plot combined metric
    bars = plt.bar(model_names, combined_scores, color=sns.color_palette("viridis", len(model_names)))
    
    # Add values on top of bars
    for bar, score in zip(bars, combined_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.2f}', ha='center', va='bottom')
    
    plt.ylim(0, 1.1)
    plt.xlabel("Algorithm")
    plt.ylabel("Combined Performance Score (0-1)")
    plt.title("Overall Performance Score (Higher is Better)")
    plt.grid(axis='y')
    plt.savefig("combined_score_comparison.png", dpi=300)
    print("✅ Combined score comparison saved as 'combined_score_comparison.png'")

def main():
    """
    Main function to run evaluations
    """
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # List of models to evaluate
    models_to_evaluate = ["SAC", "PPO", "DDPG"]#, "TD3"]
    
    # Custom Soft Actor-Critic agent
    sac_agent = SACAgent(STATE_DIM, ACTION_DIM, MAX_ACTION)
    
    # List to store results
    results = []
    
    # Evaluate each model
    for model_name in models_to_evaluate:
        if model_name == "SAC":
            # Evaluate custom SAC
            result = evaluate_model(model_name, agent=sac_agent, num_episodes=NUM_EPISODES, visualize=False)
        else:
            # Evaluate SB3 models
            if SB3_AVAILABLE:
                result = evaluate_model(model_name, num_episodes=NUM_EPISODES, visualize=False)
            else:
                print(f"Skipping {model_name} as SB3 is not available")
                continue
                
        if result:
            results.append(result)
            # Save individual model results
            np.save(f"results/{model_name}_metrics.npy", result)
    
    # Visualize comparison if we have results
    if results:
        visualize_comparison(results)
    else:
        print("No results to visualize. Check model paths and imports.")

if __name__ == "__main__":
    main()