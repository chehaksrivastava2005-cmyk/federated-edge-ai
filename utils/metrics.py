
# utils/metrics.py
"""
METRICS TRACKER - Records all training statistics across federated rounds.

Stores per-round:
  - Accuracy & loss
  - Communication cost (bytes)
  - Number of rejected clients
  - Wall-clock time

Used by:
  - plotter.py for visualization
  - results/ JSON output files
"""
import time

class MetricsTracker:
    def __init__(self, app_name):
        self.app_name = app_name
        self.rounds = []
        self.start_time = time.time()

    def log_round(self, round_num, result, elapsed):
        self.rounds.append({
            'round': round_num,
            'accuracy': result.get('accuracy', 0),
            'loss': result.get('loss', 0),
            'comm_bytes': result.get('comm_bytes', 0),
            'rejected': len(result.get('rejected_clients', [])),
            'elapsed': elapsed,
        })

    def get_summary(self):
        if not self.rounds:
            return {}
        accs = [r['accuracy'] for r in self.rounds]
        losses = [r['loss'] for r in self.rounds]
        comms = [r['comm_bytes'] for r in self.rounds]
        total_time = time.time() - self.start_time
        baseline_bytes = comms[0] if comms else 1
        return {
            'app_name': self.app_name,
            'num_rounds': len(self.rounds),
            'final_accuracy': accs[-1],
            'best_accuracy': max(accs),
            'final_loss': losses[-1],
            'total_comm_bytes': sum(comms),
            'avg_comm_bytes_per_round': sum(comms)/len(comms),
            'comm_reduction_pct': max(0, (1 - comms[-1]/max(baseline_bytes,1))*100),
            'total_rejections': sum(r['rejected'] for r in self.rounds),
            'total_time_sec': total_time,
            'round_history': self.rounds,
        }
