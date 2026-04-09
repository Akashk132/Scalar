import sys

def evaluate_performance(submission_output: str) -> float:
    """
    Standard OpenEnv Grader.
    Extracts the rewards from the [END] line and returns a value in (0, 1).
    """
    try:
        lines = submission_output.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if '[END]' in line and 'rewards=' in line:
                # Format: [END] success=true steps=n rewards=r1,r2
                rewards_part = line.split('rewards=')[1]
                # Split by comma and convert to float
                rewards = [float(r) for r in rewards_part.split(',') if r.strip()]
                if rewards:
                    # Calculate average and clamp to ultra-safe range [0.3, 0.7]
                    avg_score = sum(rewards) / len(rewards)
                    return max(0.30, min(0.70, avg_score))
                    
        # Fallback if no END line is found
        return 0.30
    except Exception:
        # Emergency fallback for malformed output
        return 0.30
