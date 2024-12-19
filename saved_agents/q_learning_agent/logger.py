import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class TrainingLogger:
    def __init__(self, log_dir="logs"):
        # Create a unique directory name with timestamp
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = os.path.join(log_dir, current_time)
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"Tensorboard logs will be saved to: {log_dir}")
        
    def log_metrics(self, metrics, step):
        """Log training metrics for a single step."""
        for name, value in metrics.items():
            self.writer.add_scalar(f'training/{name}', value, step)
    
    def log_episode_metrics(self, metrics, episode):
        """Log metrics for a complete episode."""
        for name, value in metrics.items():
            self.writer.add_scalar(f'episode/{name}', value, episode)
    
    def close(self):
        """Close the Tensorboard writer."""
        self.writer.close()
