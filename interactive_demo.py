import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from maze_environment import MazeEnvironment
from dqn_agent import DQNAgent
from advanced_visualizer import AdvancedMazeVisualizer
import os
import time

def step_by_step_solution():
    """Show the agent solving the maze step by step with detailed information."""
    print("🤖 DQN Maze Solver - Step-by-Step Demo")
    print("=" * 50)
    
    # Create environment
    env = MazeEnvironment(width=6, height=6, wall_probability=0.15)
    
    # Load trained agent
    if not os.path.exists('dqn_fast.pth'):
        print("❌ No trained model found! Please run train_fast.py first.")
        return
    
    agent = DQNAgent(state_size=env.observation_space, action_size=env.action_space)
    agent.load('dqn_fast.pth')
    print("✅ Trained agent loaded successfully!")
    
    # Create visualizer
    visualizer = AdvancedMazeVisualizer(env, agent)
    
    print(f"\n📍 Maze Configuration:")
    print(f"   Size: {env.width}x{env.height}")
    print(f"   Walls: {np.sum(env.maze)}")
    print(f"   Start: {env.start_pos}")
    print(f"   Goal: {env.goal_pos}")
    
    # Reset environment and start solving
    state = env.reset()
    path = [env.current_pos]
    total_reward = 0
    step_count = 0
    
    action_names = ['UP ↑', 'RIGHT →', 'DOWN ↓', 'LEFT ←']
    
    print(f"\n🚀 Starting solution...")
    print("-" * 50)
    
    # Get Q-values for all states to show policy
    policy = visualizer.get_learned_policy()
    
    print(f"📊 Learned Policy Preview:")
    for i, ((y, x), action) in enumerate(list(policy.items())[:5]):
        action_symbol = visualizer.action_symbols[action]
        print(f"   State {(y,x)} → {action_symbol} ({action_names[action]})")
    print(f"   ... and {len(policy)-5} more states")
    
    print(f"\n🎯 Solving the maze:")
    print("-" * 30)
    
    for step in range(50):  # Max steps
        # Get action from agent
        action = agent.act(state, eps=0.0)
        
        # Take step
        next_state, reward, done, _ = env.step(action)
        path.append(env.current_pos)
        total_reward += reward
        step_count += 1
        
        # Display step information
        print(f"Step {step_count:2d}: {action_names[action]:8s} → {env.current_pos} "
              f"(reward: {reward:+6.3f}, total: {total_reward:+7.3f})")
        
        state = next_state
        
        if done:
            print(f"\n🎉 SUCCESS! Goal reached in {step_count} steps!")
            print(f"🏆 Final score: {total_reward:.3f}")
            break
        
        # Small delay for dramatic effect
        time.sleep(0.1)
    else:
        print(f"\n⏰ Time limit reached after {step_count} steps")
        print(f"📍 Final position: {env.current_pos}")
        print(f"🎯 Goal position: {env.goal_pos}")
    
    # Show path summary
    print(f"\n📍 Complete path ({len(path)} positions):")
    path_str = " → ".join([str(pos) for pos in path])
    if len(path_str) > 60:
        path_str = path_str[:60] + "..."
    print(f"   {path_str}")
    
    # Create visualizations
    print(f"\n🖼️  Creating visualizations...")
    
    # Create comprehensive visualization with current position
    visualizer.create_comprehensive_visualization(
        current_pos=env.current_pos, 
        save_path='step_by_step_result.png'
    )
    
    print(f"✅ Visualization saved as 'step_by_step_result.png'")
    
    # Analyze the solution
    print(f"\n📊 Solution Analysis:")
    print(f"   Steps taken: {len(path) - 1}")
    print(f"   Efficiency: {'Optimal' if len(path) - 1 <= 12 else 'Good' if len(path) - 1 <= 15 else 'Could improve'}")
    print(f"   Success rate: {'100%' if done else '0%'}")
    print(f"   Final reward: {total_reward:.3f}")
    
    return path, total_reward, done

def analyze_learned_policy():
    """Analyze and display the learned policy in detail."""
    print("\n🧠 Deep Policy Analysis")
    print("=" * 30)
    
    # Load environment and agent
    env = MazeEnvironment(width=6, height=6, wall_probability=0.15)
    agent = DQNAgent(state_size=env.observation_space, action_size=env.action_space)
    agent.load('dqn_fast.pth')
    
    visualizer = AdvancedMazeVisualizer(env, agent)
    policy = visualizer.get_learned_policy()
    
    # Analyze policy by regions
    regions = {
        'Top row': [(0, x) for x in range(env.width) if (0, x) in policy],
        'Bottom row': [(env.height-1, x) for x in range(env.width) if (env.height-1, x) in policy],
        'Left column': [(y, 0) for y in range(env.height) if (y, 0) in policy],
        'Right column': [(y, env.width-1) for y in range(env.height) if (y, env.width-1) in policy],
    }
    
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # UP, RIGHT, DOWN, LEFT
    
    for region_name, positions in regions.items():
        if positions:
            print(f"\n📍 {region_name}:")
            for pos in positions[:3]:  # Show first 3
                if pos in policy:
                    action = policy[pos]
                    action_counts[action] += 1
                    symbol = visualizer.action_symbols[action]
                    print(f"   {pos} → {symbol}")
            if len(positions) > 3:
                print(f"   ... and {len(positions) - 3} more")
    
    print(f"\n📊 Action Distribution:")
    action_names = ['UP ↑', 'RIGHT →', 'DOWN ↓', 'LEFT ←']
    for action, count in action_counts.items():
        percentage = (count / len(policy)) * 100 if policy else 0
        print(f"   {action_names[action]:8s}: {count:2d} states ({percentage:4.1f}%)")
    
    print(f"\n🎯 Policy Insights:")
    most_common = max(action_counts.items(), key=lambda x: x[1])
    print(f"   Most common action: {action_names[most_common[0]]} ({most_common[1]} times)")
    print(f"   Total learned states: {len(policy)}")
    print(f"   Coverage: {len(policy)}/{env.width * env.height - np.sum(env.maze)} free spaces")

def main():
    """Run the complete interactive demo."""
    try:
        # Step-by-step solution
        path, reward, success = step_by_step_solution()
        
        # Policy analysis
        analyze_learned_policy()
        
        print(f"\n🎯 Demo Complete!")
        print(f"   Files created: step_by_step_result.png")
        print(f"   Solution: {'SUCCESS' if success else 'INCOMPLETE'}")
        print(f"   Path length: {len(path) - 1 if path else 0} steps")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("Make sure you have run 'python train_fast.py' first!")

if __name__ == "__main__":
    main() 