"""
scheduler.py — Daily Ingestion Scheduler (APScheduler).

Triggers the ingestion pipeline once per day at midnight (configurable).
Run this process in the background to keep the vector DB fresh daily.

Usage:
    python scheduler.py
"""

import logging
import sys
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from config import SCHEDULER_HOUR, SCHEDULER_MINUTE
from src.ingest import run_ingestion

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Scheduled Job ──────────────────────────────────────────────────────────────

def scheduled_ingestion_job():
    """Wrapper function invoked by APScheduler on the daily cron trigger."""
    logger.info("[SCHEDULER] Daily ingestion job triggered.")
    try:
        result = run_ingestion()
        logger.info(f"[SCHEDULER] Job complete: {result}")
    except Exception as e:
        logger.error(f"[SCHEDULER] Ingestion job failed: {e}", exc_info=True)


# ── Scheduler Entry Point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    scheduler = BlockingScheduler(timezone="Asia/Kolkata")

    scheduler.add_job(
        func=scheduled_ingestion_job,
        trigger=CronTrigger(hour=SCHEDULER_HOUR, minute=SCHEDULER_MINUTE),
        id="daily_ingestion",
        name="Daily SBI MF Document Ingestion",
        replace_existing=True,
    )

    logger.info(
        f"[SCHEDULER] Scheduled daily ingestion at "
        f"{SCHEDULER_HOUR:02d}:{SCHEDULER_MINUTE:02d} IST. "
        "Press Ctrl+C to stop."
    )

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("[SCHEDULER] Scheduler stopped.")
