import logging
import os
from datetime import datetime

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

now = datetime.now()

today_folder = now.strftime("%Y-%m-%d")

daily_log_dir = os.path.join(log_dir, today_folder)

os.makedirs(daily_log_dir, exist_ok=True)

# One log file per run named with the time the run started.
_run_started_at = now.strftime("%Y%m%d_%H%M%S")
run_log_file = os.path.join(daily_log_dir, f"run_{_run_started_at}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(run_log_file, encoding="utf-8", delay=True),
        logging.StreamHandler(),
    ],
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
