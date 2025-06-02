import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import torch
from maze_environment import MazeEnvironment
from dqn_agent import DQNAgent
import os

class AdvancedMazeVisualizer:
    def __init__(self, env, agent=None):
        """
        Advanced visualizer for DQN maze solver showing comprehensive RL information.
        
        Args:
            env: MazeEnvironment instance
            agent: Trained DQNAgent (optional)
        """
        self.env = env
        self.agent = agent
        self.action_names = ['↑', '→', '↓', '←']  # Up, Right, Down, Left
        self.action_symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}
        
    def get_learned_policy(self):
        """Extract the learned policy for each state."""
        if self.agent is None:
            return None
            
        policy = {}
        for y in range(self.env.height):
            for x in range(self.env.width):
                if self.env.maze[y, x] == 0:  # Only for free spaces
                    # Create state representation
                    state = np.zeros((self.env.height, self.env.width))
                    state[y, x] = 1
                    state_flat = state.flatten()
                    
                    # Get Q-values from agent
                    state_tensor = torch.from_numpy(state_flat).float().unsqueeze(0)
                    self.agent.qnetwork_local.eval()
                    with torch.no_grad():
                        q_values = self.agent.qnetwork_local(state_tensor)
                    
                    # Get best action
                    best_action = np.argmax(q_values.cpu().data.numpy())
                    policy[(y, x)] = best_action
                    
        return policy
    
    def create_comprehensive_visualization(self, current_pos=None, save_path='comprehensive_viz.png'):
        """Create a 4-panel comprehensive visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Panel 1: Maze State
        self._plot_maze_state(ax1, current_pos)
        
        # Panel 2: Rewards Table
        self._plot_rewards_table(ax2)
        
        # Panel 3: Learned Policy Table
        self._plot_policy_table(ax3)
        
        # Panel 4: Policy Arrows
        self._plot_policy_arrows(ax4)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comprehensive visualization saved as '{save_path}'")
        plt.close()
    
    def _plot_maze_state(self, ax, current_pos=None):
        """Plot the current maze state with agent position."""
        ax.set_title('Maze State', fontsize=14, fontweight='bold')
        
        # Create visual representation
        visual_maze = np.ones((self.env.height, self.env.width)) * 0.5  # Gray background
        
        # Set walls (black) and free spaces (white)
        for y in range(self.env.height):
            for x in range(self.env.width):
                if self.env.maze[y, x] == 1:  # Wall
                    visual_maze[y, x] = 0.0  # Black
                else:  # Free space
                    visual_maze[y, x] = 1.0  # White
        
        ax.imshow(visual_maze, cmap='gray', vmin=0, vmax=1)
        
        # Add grid
        for i in range(self.env.height + 1):
            ax.axhline(i - 0.5, color='black', linewidth=1)
        for i in range(self.env.width + 1):
            ax.axvline(i - 0.5, color='black', linewidth=1)
        
        # Add text representation
        for y in range(self.env.height):
            for x in range(self.env.width):
                if self.env.maze[y, x] == 1:
                    ax.text(x, y, '#', ha='center', va='center', fontsize=12, fontweight='bold')
                else:
                    ax.text(x, y, '.', ha='center', va='center', fontsize=12, color='gray')
        
        # Mark start and goal
        start_y, start_x = self.env.start_pos
        goal_y, goal_x = self.env.goal_pos
        ax.text(start_x, start_y, 'S', ha='center', va='center', fontsize=14, 
                fontweight='bold', color='green')
        ax.text(goal_x, goal_y, 'G', ha='center', va='center', fontsize=14, 
                fontweight='bold', color='red')
        
        # Mark current agent position if provided
        if current_pos:
            agent_y, agent_x = current_pos
            ax.add_patch(patches.Circle((agent_x, agent_y), 0.3, color='purple', alpha=0.7))
            ax.text(agent_x, agent_y, '🤖', ha='center', va='center', fontsize=12)
        
        ax.set_xlim(-0.5, self.env.width - 0.5)
        ax.set_ylim(-0.5, self.env.height - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()
    
    def _plot_rewards_table(self, ax):
        """Plot the rewards structure table."""
        ax.set_title('Rewards', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Create sample reward scenarios
        reward_data = [
            ['State', 'Reward'],
            [f'{self.env.goal_pos}', '+10'],
            [f'{self.env.goal_pos} → Wall', '-0.1'],
            [f'{self.env.goal_pos} → {self.env.goal_pos}', '+10'],
            ['Any → Goal', '+10'],
            ['Hit Wall/Boundary', '-0.1'],
            ['Normal Step', '-0.01']
        ]
        
        # Create table
        table = ax.table(cellText=reward_data[1:], 
                        colLabels=reward_data[0],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.6, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(reward_data[1:])):  # Skip header row in range
            for j in range(len(reward_data[0])):
                cell = table[(i, j)]
                cell.set_facecolor('#f0f0f0')
        
        # Style header row
        for j in range(len(reward_data[0])):
            cell = table[(0, j)]
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
    
    def _plot_policy_table(self, ax):
        """Plot a sample of the learned policy in table format."""
        ax.set_title('Learned Policy', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        if self.agent is None:
            ax.text(0.5, 0.5, 'No trained agent available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return
        
        policy = self.get_learned_policy()
        if not policy:
            ax.text(0.5, 0.5, 'Could not extract policy', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return
        
        # Sample some policy entries
        policy_data = [['State', 'Action']]
        sample_states = list(policy.keys())[:min(8, len(policy))]  # Show up to 8 states
        
        for state in sample_states:
            action_idx = policy[state]
            action_name = self.action_symbols[action_idx]
            policy_data.append([f'{state}', action_name])
        
        if len(policy_data) > 1:
            table = ax.table(cellText=policy_data[1:], 
                            colLabels=policy_data[0],
                            cellLoc='center',
                            loc='center',
                            colWidths=[0.6, 0.4])
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            # Style the table
            for i in range(len(policy_data[1:])):  # Skip header row in range
                for j in range(len(policy_data[0])):
                    cell = table[(i, j)]
                    cell.set_facecolor('#f0f0f0')
            
            # Style header row  
            for j in range(len(policy_data[0])):
                cell = table[(0, j)]
                cell.set_facecolor('#2196F3')
                cell.set_text_props(weight='bold', color='white')
    
    def _plot_policy_arrows(self, ax):
        """Plot policy arrows overlaid on the maze."""
        ax.set_title('Policy Arrows', fontsize=14, fontweight='bold')
        
        if self.agent is None:
            ax.text(0.5, 0.5, 'No trained agent available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return
        
        # Create background maze
        visual_maze = np.ones((self.env.height, self.env.width)) * 0.9
        
        for y in range(self.env.height):
            for x in range(self.env.width):
                if self.env.maze[y, x] == 1:  # Wall
                    visual_maze[y, x] = 0.3  # Dark gray for walls
        
        ax.imshow(visual_maze, cmap='gray', vmin=0, vmax=1)
        
        # Get policy and overlay arrows
        policy = self.get_learned_policy()
        if policy:
            for (y, x), action in policy.items():
                if (y, x) == self.env.goal_pos:
                    # Mark goal
                    ax.text(x, y, 'G', ha='center', va='center', fontsize=14, 
                           fontweight='bold', color='red')
                elif (y, x) == self.env.start_pos:
                    # Mark start with arrow
                    ax.text(x, y, 'S', ha='center', va='center', fontsize=12, 
                           fontweight='bold', color='green')
                    # Add small arrow
                    ax.text(x + 0.2, y - 0.2, self.action_symbols[action], 
                           ha='center', va='center', fontsize=10, color='blue')
                else:
                    # Add policy arrow
                    ax.text(x, y, self.action_symbols[action], ha='center', va='center', 
                           fontsize=16, fontweight='bold', color='blue')
        
        # Add walls
        for y in range(self.env.height):
            for x in range(self.env.width):
                if self.env.maze[y, x] == 1:
                    ax.text(x, y, '#', ha='center', va='center', fontsize=12, 
                           fontweight='bold', color='black')
        
        ax.set_xlim(-0.5, self.env.width - 0.5)
        ax.set_ylim(-0.5, self.env.height - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()
    
    def create_animated_solution(self, save_path='animated_solution.png'):
        """Create an animated visualization showing the agent solving the maze."""
        if self.agent is None:
            print("No trained agent available for animation")
            return
        
        # Run the agent and collect path
        state = self.env.reset()
        path = [self.env.current_pos]
        
        for step in range(50):  # Max steps
            action = self.agent.act(state, eps=0.0)
            next_state, reward, done, _ = self.env.step(action)
            path.append(self.env.current_pos)
            state = next_state
            
            if done:
                break
        
        # Create visualization for final state
        self.create_comprehensive_visualization(current_pos=self.env.current_pos, 
                                              save_path=save_path)
        
        return path

def main():
    """Demonstrate the advanced visualization."""
    print("Creating Advanced DQN Maze Visualizations")
    print("=========================================")
    
    # Create environment
    env = MazeEnvironment(width=6, height=6, wall_probability=0.15)
    
    # Load trained agent if available
    agent = None
    if os.path.exists('dqn_fast.pth'):
        print("Loading trained agent...")
        agent = DQNAgent(state_size=env.observation_space, action_size=env.action_space)
        agent.load('dqn_fast.pth')
        print("Agent loaded successfully!")
    else:
        print("No trained agent found. Showing maze structure only.")
    
    # Create visualizer
    visualizer = AdvancedMazeVisualizer(env, agent)
    
    # Generate comprehensive visualization
    visualizer.create_comprehensive_visualization(save_path='advanced_maze_viz.png')
    
    # If agent is available, create animated solution
    if agent:
        path = visualizer.create_animated_solution('solution_animation.png')
        print(f"Solution path length: {len(path) - 1} steps")
        print(f"Path: {' → '.join([str(pos) for pos in path[:5]])}{'...' if len(path) > 5 else ''}")
    
    print("\nAdvanced visualizations created!")
    print("Files generated:")
    print("- advanced_maze_viz.png (4-panel comprehensive view)")
    if agent:
        print("- solution_animation.png (agent solution)")

if __name__ == "__main__":
    main() 