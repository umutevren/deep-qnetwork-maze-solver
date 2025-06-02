import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from maze_environment import MazeEnvironment
from dqn_agent import DQNAgent
import os

def demo_trained_agent_fast(model_path='dqn_fast.pth', maze_size=6):
    """
    Demonstrate a trained DQN agent solving mazes (fast/non-interactive version).
    
    Args:
        model_path: Path to the trained model
        maze_size: Size of the maze to solve
    """
    print("DQN Maze Solver Demo (Fast)")
    print("===========================")
    
    # Create environment
    env = MazeEnvironment(width=maze_size, height=maze_size, wall_probability=0.2)
    
    # Create agent
    state_size = env.observation_space
    action_size = env.action_space
    
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        epsilon=0.0  # No exploration for demo
    )
    
    # Load trained model if it exists
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}")
        agent.load(model_path)
        print("Model loaded successfully!")
    else:
        print(f"No trained model found at {model_path}")
        print("The agent will act randomly. Train the model first using train_fast.py")
        return
    
    print(f"\nGenerated {maze_size}x{maze_size} maze with {np.sum(env.maze)} walls")
    print(f"Start: {env.start_pos}, Goal: {env.goal_pos}")
    
    # Run multiple test episodes
    total_episodes = 5
    successes = 0
    all_paths = []
    all_scores = []
    
    for episode in range(total_episodes):
        print(f"\n--- Test Episode {episode + 1} ---")
        
        # Reset environment
        state = env.reset()
        path = [env.current_pos]
        total_reward = 0
        steps = 0
        max_steps = 50
        
        action_names = ['Up', 'Right', 'Down', 'Left']
        
        for step in range(max_steps):
            action = agent.act(state, eps=0.0)  # No exploration
            next_state, reward, done, _ = env.step(action)
            
            path.append(env.current_pos)
            total_reward += reward
            steps += 1
            
            print(f"  Step {step + 1}: {action_names[action]} -> {env.current_pos}")
            
            state = next_state
            
            if done:
                print(f"  ✅ GOAL REACHED in {steps} steps!")
                successes += 1
                break
        else:
            print(f"  ❌ Failed to reach goal in {max_steps} steps")
        
        print(f"  Total reward: {total_reward:.3f}")
        print(f"  Path length: {len(path) - 1}")
        
        all_paths.append(path)
        all_scores.append(total_reward)
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Success rate: {successes}/{total_episodes} ({100*successes/total_episodes:.1f}%)")
    print(f"Average score: {np.mean(all_scores):.2f}")
    if successes > 0:
        successful_paths = [len(path)-1 for i, path in enumerate(all_paths) if all_scores[i] > 5]
        print(f"Average successful path length: {np.mean(successful_paths):.1f} steps")
    
    # Save visualization of the best episode
    if all_scores:
        best_episode = np.argmax(all_scores)
        best_path = all_paths[best_episode]
        
        print(f"\nSaving visualization of best episode...")
        save_demo_visualization(env, best_path, all_scores[best_episode])

def save_demo_visualization(env, path, score):
    """Save the demo result as an image file."""
    visual_maze = env.get_maze_with_path(path)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(visual_maze, cmap='RdYlBu')
    plt.title(f'DQN Demo Solution\nSteps: {len(path) - 1}, Score: {score:.2f}')
    plt.colorbar(label='0=Free, 0.3=Start, 0.4=Path, 0.7=Goal, 1=Wall')
    
    # Add grid
    plt.xticks(range(env.width))
    plt.yticks(range(env.height))
    plt.grid(True, alpha=0.3)
    
    # Add step numbers along the path
    for i, pos in enumerate(path[::max(1, len(path)//8)]):  # Show step numbers
        plt.text(pos[1], pos[0], str(i), fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.savefig('demo_result.png', dpi=100, bbox_inches='tight')
    print("Demo visualization saved as 'demo_result.png'")
    plt.close()

if __name__ == "__main__":
    demo_trained_agent_fast() 