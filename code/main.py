import logging
from logging.handlers import RotatingFileHandler
import os
from loader import cfg

log_file = '../outputs/run.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)

handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[handler, logging.StreamHandler()]
)
logging.info("Logging initialized (INFO level, with file rotation).")


import argparse
from workflow import run_workflow

def main():
    # --- CLI Argument Setup ---
    parser = argparse.ArgumentParser(description="Multi-objective GP for Regression/Classification")
    parser.add_argument('--task', choices=['regression', 'classification'], required=True,
                        help="Specify the task: 'regression' or 'classification'")
    parser.add_argument('--dataset', required=True,
                        help="Dataset name, e.g., 'diabetes', 'iris'")
    args = parser.parse_args()

    logging.info(f"Running workflow task: {args.task}, Dataset: {args.dataset}")
    run_workflow(args.dataset, cfg, task=args.task)

if __name__ == "__main__":
    main()
