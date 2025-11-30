"""
Logging utilities for RL experiments.
"""
import csv
from pathlib import Path
from datetime import datetime


class Logger:
    """Logger for training metrics."""
    
    def __init__(self, log_dir, experiment_name):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.csv"
        
        # Write header
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'reward', 'steps', 'epsilon'])
        
        print(f"Logging to: {self.log_file}")
    
    def log(self, episode, reward, steps, epsilon):
        """
        Log episode metrics.
        
        Args:
            episode: Episode number
            reward: Total episode reward
            steps: Number of steps
            epsilon: Current epsilon value
        """
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, reward, steps, epsilon])


class TensorBoardLogger:
    """TensorBoard logger (optional)."""
    
    def __init__(self, log_dir, experiment_name):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            self.log_dir = Path(log_dir) / "tensorboard"
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.writer = SummaryWriter(self.log_dir / f"{experiment_name}_{timestamp}")
            self.enabled = True
            print(f"TensorBoard logging enabled: {self.log_dir}")
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
            self.enabled = False
    
    def log_scalar(self, tag, value, step):
        """Log scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag, tag_scalar_dict, step):
        """Log multiple scalars."""
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def close(self):
        """Close writer."""
        if self.enabled:
            self.writer.close()
