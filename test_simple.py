import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from maze_environment import MazeEnvironment
from dqn_agent import DQNAgent
import os

def test_on_similar_maze():
    """Test the trained agent on a maze similar to training conditions."""
    print("Testing DQN on Similar Maze Configuration")
    print("========================================")
    
    # Use the same configuration as training (low wall probability)
    env = MazeEnvironment(width=6, height=6, wall_probability=0.15)
    
    # Create agent
    state_size = env.observation_space
    action_size = env.action_space
    
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        epsilon=0.0  # No exploration for demo
    )
    
    # Load trained model
    if os.path.exists('dqn_fast.pth'):
        print("Loading trained model...")
        agent.load('dqn_fast.pth')
        print("Model loaded successfully!")
    else:
        print("No trained model found!")
        return
    
    print(f"\nGenerated {env.width}x{env.height} maze with {np.sum(env.maze)} walls")
    print(f"Start: {env.start_pos}, Goal: {env.goal_pos}")
    
    # Test multiple episodes
    total_episodes = 5
    successes = 0
    all_results = []
    
    for episode in range(total_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        
        # Reset environment
        state = env.reset()
        path = [env.current_pos]
        total_reward = 0
        max_steps = 30
        
        action_names = ['Up', 'Right', 'Down', 'Left']
        
        for step in range(max_steps):
            action = agent.act(state, eps=0.0)
            next_state, reward, done, _ = env.step(action)
            
            path.append(env.current_pos)
            total_reward += reward
            
            print(f"  Step {step + 1}: {action_names[action]} -> {env.current_pos} (reward: {reward:.3f})")
            
            state = next_state
            
            if done:
                print(f"  ✅ SUCCESS! Reached goal in {len(path) - 1} steps")
                successes += 1
                break
        else:
            print(f"  ❌ Failed to reach goal in {max_steps} steps")
        
        print(f"  Total reward: {total_reward:.3f}")
        all_results.append((len(path) - 1, total_reward, done))
        
        # Save visualization for successful episodes
        if done:
            save_episode_viz(env, path, episode + 1, total_reward)
    
    # Summary
    print(f"\n=== FINAL RESULTS ===")
    print(f"Success rate: {successes}/{total_episodes} ({100*successes/total_episodes:.1f}%)")
    
    if successes > 0:
        successful_results = [r for r in all_results if r[2]]  # r[2] is 'done' flag
        avg_steps = np.mean([r[0] for r in successful_results])
        avg_score = np.mean([r[1] for r in successful_results])
        print(f"Average successful path length: {avg_steps:.1f} steps")
        print(f"Average successful score: {avg_score:.2f}")
    
    print(f"Files saved: episode visualization PNGs")

def save_episode_viz(env, path, episode_num, score):
    """Save visualization for an episode."""
    visual_maze = env.get_maze_with_path(path)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(visual_maze, cmap='RdYlBu')
    plt.title(f'Episode {episode_num} - SUCCESS\nSteps: {len(path) - 1}, Score: {score:.2f}')
    plt.colorbar(label='0=Free, 0.3=Start, 0.4=Path, 0.7=Goal, 1=Wall')
    
    # Add grid
    plt.xticks(range(env.width))
    plt.yticks(range(env.height))
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f'episode_{episode_num}_success.png', dpi=100, bbox_inches='tight')
    print(f"    Saved: episode_{episode_num}_success.png")
    plt.close()

if __name__ == "__main__":
    test_on_similar_maze() 