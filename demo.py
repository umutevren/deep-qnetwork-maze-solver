import numpy as np
import matplotlib.pyplot as plt
from maze_environment import MazeEnvironment
from dqn_agent import DQNAgent
import os

def demo_trained_agent(model_path='dqn_final.pth', maze_size=8):
    """
    Demonstrate a trained DQN agent solving mazes.
    
    Args:
        model_path: Path to the trained model
        maze_size: Size of the maze to solve
    """
    print("DQN Maze Solver Demo")
    print("===================")
    
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
    else:
        print(f"No trained model found at {model_path}")
        print("The agent will act randomly. Train the model first using train_dqn.py")
    
    # Show the maze
    print(f"\nGenerated {maze_size}x{maze_size} maze:")
    plt.figure(figsize=(6, 6))
    plt.imshow(env.maze, cmap='RdYlBu')
    plt.title('Maze Layout')
    plt.colorbar(label='0=Free path, 1=Wall')
    
    # Mark start and goal
    plt.scatter(env.start_pos[1], env.start_pos[0], c='green', s=100, marker='s', label='Start')
    plt.scatter(env.goal_pos[1], env.goal_pos[0], c='red', s=100, marker='*', label='Goal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Run the agent
    print(f"\nRunning DQN agent...")
    state = env.reset()
    path = [env.current_pos]
    total_reward = 0
    steps = 0
    max_steps = 100
    
    action_names = ['Up', 'Right', 'Down', 'Left']
    
    for step in range(max_steps):
        action = agent.act(state, eps=0.0)  # No exploration
        next_state, reward, done, _ = env.step(action)
        
        path.append(env.current_pos)
        total_reward += reward
        steps += 1
        
        print(f"Step {step + 1}: {action_names[action]} -> Position {env.current_pos}, Reward: {reward:.3f}")
        
        state = next_state
        
        if done:
            print(f"\nGoal reached in {steps} steps!")
            break
    else:
        print(f"\nMax steps ({max_steps}) reached without finding goal")
    
    print(f"Total reward: {total_reward:.3f}")
    print(f"Path length: {len(path) - 1}")
    
    # Visualize the solution
    if len(path) > 1:
        visual_maze = env.get_maze_with_path(path)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(visual_maze, cmap='RdYlBu')
        plt.title(f'DQN Solution Path\n(Steps: {len(path) - 1}, Reward: {total_reward:.2f})')
        plt.colorbar(label='0=Free, 0.3=Start, 0.4=Path, 0.7=Goal, 1=Wall')
        
        # Add grid and labels
        plt.xticks(range(env.width))
        plt.yticks(range(env.height))
        plt.grid(True, alpha=0.3)
        
        # Add step numbers along the path
        for i, pos in enumerate(path[::max(1, len(path)//10)]):  # Show every few steps
            plt.text(pos[1], pos[0], str(i), fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        plt.show()

def compare_random_vs_trained():
    """Compare random agent vs trained agent performance."""
    env = MazeEnvironment(width=6, height=6, wall_probability=0.15)
    
    # Test random agent
    print("Testing random agent...")
    random_scores = []
    for _ in range(10):
        state = env.reset()
        score = 0
        for _ in range(50):
            action = np.random.randint(0, 4)
            _, reward, done, _ = env.step(action)
            score += reward
            if done:
                break
        random_scores.append(score)
    
    print(f"Random agent average score: {np.mean(random_scores):.2f} ± {np.std(random_scores):.2f}")
    
    # Test trained agent (if available)
    if os.path.exists('dqn_final.pth'):
        agent = DQNAgent(state_size=env.observation_space, action_size=env.action_space, epsilon=0.0)
        agent.load('dqn_final.pth')
        
        trained_scores = []
        for _ in range(10):
            state = env.reset()
            score = 0
            for _ in range(50):
                action = agent.act(state, eps=0.0)
                state, reward, done, _ = env.step(action)
                score += reward
                if done:
                    break
            trained_scores.append(score)
        
        print(f"Trained agent average score: {np.mean(trained_scores):.2f} ± {np.std(trained_scores):.2f}")
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        plt.boxplot([random_scores, trained_scores], labels=['Random Agent', 'Trained DQN'])
        plt.title('Performance Comparison')
        plt.ylabel('Episode Score')
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print("No trained model found for comparison")

if __name__ == "__main__":
    print("Choose demo option:")
    print("1. Run trained agent on a maze")
    print("2. Compare random vs trained agent")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        demo_trained_agent()
    elif choice == "2":
        compare_random_vs_trained()
    else:
        print("Invalid choice. Running default demo...")
        demo_trained_agent() 