import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from collections import deque
import torch
import time
from maze_environment import MazeEnvironment
from dqn_agent import DQNAgent

def train_dqn_fast(env, agent, n_episodes=200, max_steps=50, solve_score=5.0, 
                   print_every=25):
    """
    Fast training for demonstration purposes.
    
    Args:
        env: Environment
        agent: DQN agent
        n_episodes: Maximum number of training episodes (reduced)
        max_steps: Maximum steps per episode (reduced)
        solve_score: Score threshold for considering the problem solved
        print_every: How often to print progress
    
    Returns:
        scores: List of scores from each episode
        episode_lengths: List of episode lengths
    """
    scores = []
    episode_lengths = []
    scores_window = deque(maxlen=50)  # Smaller window
    
    print("Training DQN agent (fast mode)...")
    
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        steps = 0
        
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            steps += 1
            
            if done:
                break
        
        scores_window.append(score)
        scores.append(score)
        episode_lengths.append(steps)
        
        # Print progress
        if i_episode % print_every == 0:
            print(f'Episode {i_episode:3d}\t'
                  f'Average Score: {np.mean(scores_window):6.2f}\t'
                  f'Epsilon: {agent.epsilon:.3f}\t'
                  f'Steps: {steps:2d}')
        
        # Check if environment is solved
        if np.mean(scores_window) >= solve_score and i_episode >= 50:
            print(f'\nEnvironment solved in {i_episode} episodes!\t'
                  f'Average Score: {np.mean(scores_window):.2f}')
            break
    
    return scores, episode_lengths

def test_agent_fast(env, agent, n_episodes=3):
    """
    Quick test of the trained agent.
    
    Args:
        env: Environment
        agent: Trained DQN agent
        n_episodes: Number of test episodes
    
    Returns:
        scores: List of test scores
        paths: List of paths taken by the agent
    """
    scores = []
    paths = []
    
    print("\nTesting trained agent...")
    
    for episode in range(n_episodes):
        state = env.reset()
        score = 0
        path = [env.current_pos]
        
        for step in range(50):  # Max steps
            action = agent.act(state, eps=0.0)  # No exploration
            next_state, reward, done, _ = env.step(action)
            
            path.append(env.current_pos)
            state = next_state
            score += reward
            
            if done:
                print(f"Test episode {episode + 1}: SOLVED in {len(path) - 1} steps! Score: {score:.2f}")
                break
        else:
            print(f"Test episode {episode + 1}: Failed to reach goal in {len(path) - 1} steps. Score: {score:.2f}")
        
        scores.append(score)
        paths.append(path)
    
    return scores, paths

def save_training_plots(scores, episode_lengths):
    """Save training results as image files."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot scores
    ax1.plot(scores)
    ax1.set_title('Training Scores')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.grid(True)
    
    # Plot moving average
    window_size = min(20, len(scores))
    if len(scores) >= window_size:
        moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(scores)), moving_avg, 'r-', 
                label=f'{window_size}-episode moving average')
        ax1.legend()
    
    # Plot episode lengths
    ax2.plot(episode_lengths)
    ax2.set_title('Episode Lengths')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=100, bbox_inches='tight')
    print("Training plots saved as 'training_results.png'")
    plt.close()

def save_solution_visualization(env, path):
    """Save the solution path as an image file."""
    visual_maze = env.get_maze_with_path(path)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(visual_maze, cmap='RdYlBu')
    plt.title(f'DQN Solution Path\n(Steps: {len(path) - 1})')
    plt.colorbar(label='0=Free, 0.3=Start, 0.4=Path, 0.7=Goal, 1=Wall')
    
    # Add grid
    plt.xticks(range(env.width))
    plt.yticks(range(env.height))
    plt.grid(True, alpha=0.3)
    
    plt.savefig('solution_path.png', dpi=100, bbox_inches='tight')
    print("Solution visualization saved as 'solution_path.png'")
    plt.close()

def save_maze_layout(env):
    """Save the maze layout as an image file."""
    plt.figure(figsize=(6, 6))
    plt.imshow(env.maze, cmap='RdYlBu')
    plt.title('Maze Layout')
    plt.colorbar(label='0=Free path, 1=Wall')
    
    # Mark start and goal
    plt.scatter(env.start_pos[1], env.start_pos[0], c='green', s=100, marker='s', label='Start')
    plt.scatter(env.goal_pos[1], env.goal_pos[0], c='red', s=100, marker='*', label='Goal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('maze_layout.png', dpi=100, bbox_inches='tight')
    print("Maze layout saved as 'maze_layout.png'")
    plt.close()

def main():
    """Main fast training and testing function."""
    print("DQN Maze Solver - Fast Demo")
    print("===========================")
    
    # Create a smaller environment for faster training
    env = MazeEnvironment(width=6, height=6, wall_probability=0.15)
    
    print(f"Generated {env.width}x{env.height} maze with {np.sum(env.maze)} walls")
    print(f"Start: {env.start_pos}, Goal: {env.goal_pos}")
    
    # Save maze layout
    save_maze_layout(env)
    
    # Create agent with faster learning parameters
    state_size = env.observation_space
    action_size = env.action_space
    
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=0.005,           # Higher learning rate for faster learning
        buffer_size=5000,   # Smaller buffer
        batch_size=32,      # Smaller batch size
        gamma=0.95,         # Slightly lower discount factor
        tau=0.01,           # Faster target network updates
        epsilon=1.0,
        epsilon_min=0.05,   # Higher minimum exploration
        epsilon_decay=0.99, # Faster epsilon decay
        update_freq=2       # More frequent updates
    )
    
    print(f"State size: {state_size}, Action size: {action_size}")
    
    # Train the agent
    start_time = time.time()
    scores, episode_lengths = train_dqn_fast(
        env=env,
        agent=agent,
        n_episodes=200,
        max_steps=50,
        solve_score=7.0,  # Reasonable score for 6x6 maze
        print_every=25
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Save training plots
    save_training_plots(scores, episode_lengths)
    
    # Test the trained agent
    test_scores, test_paths = test_agent_fast(env, agent, n_episodes=3)
    
    print(f"\nTest Results Summary:")
    print(f"Average test score: {np.mean(test_scores):.2f}")
    print(f"Success rate: {sum(1 for score in test_scores if score > 5)}/{len(test_scores)}")
    
    # Visualize the best solution
    if test_paths:
        best_episode = np.argmax(test_scores)
        best_path = test_paths[best_episode]
        
        print(f"Best test episode: {len(best_path) - 1} steps")
        save_solution_visualization(env, best_path)
    
    # Save the final model
    agent.save('dqn_fast.pth')
    print("Model saved as 'dqn_fast.pth'")
    
    # Final summary
    print(f"\nFinal Results:")
    print(f"- Training episodes: {len(scores)}")
    print(f"- Final average score: {np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores):.2f}")
    print(f"- Agent epsilon: {agent.epsilon:.3f}")
    print(f"- Files created: maze_layout.png, training_results.png, solution_path.png, dqn_fast.pth")

if __name__ == "__main__":
    main() 