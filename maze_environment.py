import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import random

class MazeEnvironment:
    def __init__(self, width: int = 10, height: int = 10, wall_probability: float = 0.2):
        """
        Initialize the maze environment.
        
        Args:
            width: Width of the maze
            height: Height of the maze
            wall_probability: Probability of a cell being a wall when generating random maze
        """
        self.width = width
        self.height = height
        self.wall_probability = wall_probability
        
        # Actions: 0=up, 1=right, 2=down, 3=left
        self.action_space = 4
        self.observation_space = width * height
        
        # Generate maze
        self.maze = self._generate_maze()
        
        # Set start and goal positions
        self.start_pos = (0, 0)
        self.goal_pos = (height - 1, width - 1)
        
        # Current position
        self.current_pos = self.start_pos
        
        # Action mappings
        self.actions = {
            0: (-1, 0),  # up
            1: (0, 1),   # right
            2: (1, 0),   # down
            3: (0, -1)   # left
        }
        
    def _generate_maze(self) -> np.ndarray:
        """Generate a random maze."""
        maze = np.random.choice([0, 1], 
                               size=(self.height, self.width), 
                               p=[1-self.wall_probability, self.wall_probability])
        
        # Ensure start and goal are not walls
        maze[0, 0] = 0  # start
        maze[self.height-1, self.width-1] = 0  # goal
        
        return maze
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.current_pos = self.start_pos
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        state = np.zeros((self.height, self.width))
        state[self.current_pos[0], self.current_pos[1]] = 1  # Agent position
        state = state.flatten()
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            next_state, reward, done, info
        """
        # Calculate next position
        dy, dx = self.actions[action]
        next_y = self.current_pos[0] + dy
        next_x = self.current_pos[1] + dx
        
        # Check boundaries
        if (next_y < 0 or next_y >= self.height or 
            next_x < 0 or next_x >= self.width):
            # Hit boundary - stay in place, negative reward
            reward = -0.1
            done = False
        elif self.maze[next_y, next_x] == 1:
            # Hit wall - stay in place, negative reward
            reward = -0.1
            done = False
        else:
            # Valid move
            self.current_pos = (next_y, next_x)
            
            if self.current_pos == self.goal_pos:
                # Reached goal
                reward = 10.0
                done = True
            else:
                # Normal step
                reward = -0.01  # Small negative reward to encourage efficiency
                done = False
        
        next_state = self._get_state()
        info = {}
        
        return next_state, reward, done, info
    
    def render(self, show_agent=True):
        """Render the maze."""
        visual_maze = self.maze.copy().astype(float)
        
        if show_agent:
            visual_maze[self.current_pos[0], self.current_pos[1]] = 0.5  # Agent
        
        visual_maze[self.start_pos[0], self.start_pos[1]] = 0.3  # Start (green)
        visual_maze[self.goal_pos[0], self.goal_pos[1]] = 0.7   # Goal (blue)
        
        plt.imshow(visual_maze, cmap='RdYlBu')
        plt.title('Maze Environment')
        plt.colorbar(label='0=Free, 0.3=Start, 0.5=Agent, 0.7=Goal, 1=Wall')
        plt.show()
    
    def get_maze_with_path(self, path: List[Tuple[int, int]]) -> np.ndarray:
        """Visualize maze with a path."""
        visual_maze = self.maze.copy().astype(float)
        
        # Mark path
        for pos in path:
            if pos != self.start_pos and pos != self.goal_pos:
                visual_maze[pos[0], pos[1]] = 0.4
        
        visual_maze[self.start_pos[0], self.start_pos[1]] = 0.3  # Start
        visual_maze[self.goal_pos[0], self.goal_pos[1]] = 0.7   # Goal
        
        return visual_maze 