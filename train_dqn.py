import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch
import time
from maze_environment import MazeEnvironment
from dqn_agent import DQNAgent

def train_dqn(env, agent, n_episodes=2000, max_steps=200, solve_score=195.0, 
              print_every=100, save_every=500):
    """
    Train the DQN agent.
    
    Args:
        env: Environment
        agent: DQN agent
        n_episodes: Maximum number of training episodes
        max_steps: Maximum steps per episode
        solve_score: Score threshold for considering the problem solved
        print_every: How often to print progress
        save_every: How often to save the model
    
    Returns:
        scores: List of scores from each episode
        episode_lengths: List of episode lengths
    """
    scores = []
    episode_lengths = []
    scores_window = deque(maxlen=100)
    
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
            print(f'\rEpisode {i_episode}\t'
                  f'Average Score: {np.mean(scores_window):.2f}\t'
                  f'Epsilon: {agent.epsilon:.3f}\t'
                  f'Last Episode Steps: {steps}')
        
        # Save model
        if i_episode % save_every == 0:
            agent.save(f'dqn_checkpoint_{i_episode}.pth')
            print(f'\nModel saved at episode {i_episode}')
        
        # Check if environment is solved
        if np.mean(scores_window) >= solve_score and i_episode >= 100:
            print(f'\nEnvironment solved in {i_episode} episodes!\t'
                  f'Average Score: {np.mean(scores_window):.2f}')
            agent.save('dqn_solved.pth')
            break
    
    return scores, episode_lengths

def test_agent(env, agent, n_episodes=10, render=False):
    """
    Test the trained agent.
    
    Args:
        env: Environment
        agent: Trained DQN agent
        n_episodes: Number of test episodes
        render: Whether to render the environment
    
    Returns:
        scores: List of test scores
        paths: List of paths taken by the agent
    """
    scores = []
    paths = []
    
    for episode in range(n_episodes):
        state = env.reset()
        score = 0
        path = [env.current_pos]
        
        for step in range(200):  # Max steps
            action = agent.act(state, eps=0.0)  # No exploration
            next_state, reward, done, _ = env.step(action)
            
            path.append(env.current_pos)
            state = next_state
            score += reward
            
            if render and episode == 0:  # Render first episode
                print(f"Step {step}: Action {action}, Position {env.current_pos}, Reward {reward}")
            
            if done:
                break
        
        scores.append(score)
        paths.append(path)
        
        if render and episode == 0:
            print(f"Episode {episode + 1}: Score = {score}, Steps = {len(path) - 1}")
    
    return scores, paths

def plot_training_results(scores, episode_lengths):
    """Plot training results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot scores
    ax1.plot(scores)
    ax1.set_title('Training Scores')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.grid(True)
    
    # Plot moving average
    window_size = min(100, len(scores))
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
    plt.show()

def visualize_solution(env, path):
    """Visualize the solution path."""
    visual_maze = env.get_maze_with_path(path)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(visual_maze, cmap='RdYlBu')
    plt.title(f'DQN Solution Path (Length: {len(path) - 1} steps)')
    plt.colorbar(label='0=Free, 0.3=Start, 0.4=Path, 0.7=Goal, 1=Wall')
    
    # Add grid
    plt.xticks(range(env.width))
    plt.yticks(range(env.height))
    plt.grid(True, alpha=0.3)
    
    plt.show()

def main():
    """Main training and testing function."""
    print("Initializing maze environment and DQN agent...")
    
    # Create environment
    env = MazeEnvironment(width=8, height=8, wall_probability=0.2)
    
    # Show the maze
    print("Generated maze:")
    env.render(show_agent=False)
    
    # Create agent
    state_size = env.observation_space
    action_size = env.action_space
    
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=0.001,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        tau=0.001,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        update_freq=4
    )
    
    print(f"State size: {state_size}, Action size: {action_size}")
    print("Starting training...")
    
    # Train the agent
    start_time = time.time()
    scores, episode_lengths = train_dqn(
        env=env,
        agent=agent,
        n_episodes=1000,
        max_steps=100,
        solve_score=8.0,  # Adjust based on your maze and reward structure
        print_every=50
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Plot training results
    plot_training_results(scores, episode_lengths)
    
    # Test the trained agent
    print("\nTesting trained agent...")
    test_scores, test_paths = test_agent(env, agent, n_episodes=5, render=True)
    
    print(f"\nTest Results:")
    print(f"Average test score: {np.mean(test_scores):.2f}")
    print(f"Average path length: {np.mean([len(path)-1 for path in test_paths]):.2f}")
    
    # Visualize the best solution
    best_episode = np.argmax(test_scores)
    best_path = test_paths[best_episode]
    
    print(f"\nBest test episode path length: {len(best_path) - 1}")
    visualize_solution(env, best_path)
    
    # Save the final model
    agent.save('dqn_final.pth')
    print("Final model saved as 'dqn_final.pth'")

if __name__ == "__main__":
    main() 