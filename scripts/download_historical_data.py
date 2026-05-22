import os
import sys
import time
import logging
import fastf1

# Ensure we can import from the src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
from src.data_fetcher import fetch_gp, SESSION_TYPES, DATA_DIR, get_gp_folder_name

# Setup basic logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("HistoricalDownloader")

original_get_session = fastf1.get_session
original_load = fastf1.core.Session.load
original_get_event_schedule = fastf1.get_event_schedule

def sleep_with_countdown(seconds):
    """Sleep for a given number of seconds while displaying a live countdown."""
    for remaining in range(seconds, 0, -1):
        mins, secs = divmod(remaining, 60)
        sys.stdout.write(f"\rTime remaining until retry: {mins:02d}:{secs:02d}  ")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\r" + " " * 50 + "\r")
    sys.stdout.flush()

def handle_rate_limit(func, *args, **kwargs):
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            # FastF1 API rate limit detection
            if "429" in msg or "rate limit" in msg or "too many requests" in msg or "500 calls" in msg or "quota" in msg:
                logger.warning(f"API RATE LIMIT DETECTED! Sleeping for 1 hour (3600 seconds)...")
                logger.warning(f"Error details: {e}")
                sleep_with_countdown(3600)
                logger.info("Waking up and retrying the request...")
            else:
                # Re-raise if it's not a rate limit error (e.g. data simply doesn't exist)
                raise e

def patched_get_session(*args, **kwargs):
    return handle_rate_limit(original_get_session, *args, **kwargs)

def patched_load(self, *args, **kwargs):
    # self is the session object
    return handle_rate_limit(original_load, self, *args, **kwargs)

def patched_get_event_schedule(*args, **kwargs):
    return handle_rate_limit(original_get_event_schedule, *args, **kwargs)

# Apply patches
fastf1.get_session = patched_get_session
fastf1.core.Session.load = patched_load
fastf1.get_event_schedule = patched_get_event_schedule

# Map FastF1 session names back to our abbreviations
REVERSE_SESSION_TYPES = {v: k for k, v in SESSION_TYPES.items()}

# Map abbreviations to output filenames
FILENAME_MAP = {
    "FP1": "fp1.json",
    "FP2": "fp2.json",
    "FP3": "fp3.json",
    "Q": "qualifying.json",
    "SQ": "sprint_qualifying.json",
    "SS": "sprint_shootout.json",
    "S": "sprint.json",
    "R": "race.json"
}

def get_scheduled_sessions(event):
    """Determine which sessions are actually scheduled for this GP."""
    scheduled = []
    for i in range(1, 6):
        session_name = event.get(f'Session{i}')
        if session_name and session_name in REVERSE_SESSION_TYPES:
            scheduled.append(REVERSE_SESSION_TYPES[session_name])
    return scheduled


def main():
    logger.info("Starting F1 Historical Data Downloader")
    logger.info("Target: 2022 Season -> Present (Latest Race)")
    sessions_to_fetch = ["FP1", "FP2", "FP3", "Q", "SQ", "SS", "S", "R"]
    
    for year in range(2022, 2027):
        logger.info(f"Fetching schedule for {year} season...")
        schedule = fastf1.get_event_schedule(year)
        
        for _, event in schedule.iterrows():
            # Skip pre-season testing
            if event['EventFormat'] == 'testing':
                continue
                
            round_num = event['RoundNumber']
            gp_name = event['EventName']
            
            # Stop condition: Only download past events
            import datetime
            if event['EventDate'] > datetime.datetime.now():
                logger.info(f"✅ Reached future event: {year} {gp_name}. Historical data download complete up to today!")
                return
                
            # Determine which sessions are actually scheduled
            scheduled_sessions = get_scheduled_sessions(event)
            if not scheduled_sessions:
                # Fallback to standard if we couldn't parse the event sessions
                scheduled_sessions = ["FP1", "FP2", "FP3", "Q", "SQ", "SS", "S", "R"]
                
            # Check which sessions we already have
            gp_folder = get_gp_folder_name(round_num, gp_name)
            gp_path = Path(DATA_DIR) / str(year) / gp_folder
            
            sessions_to_fetch = []
            for session in scheduled_sessions:
                file_path = gp_path / FILENAME_MAP.get(session, f"{session.lower()}.json")
                # If file doesn't exist or is empty, we need to fetch it
                if not file_path.exists() or file_path.stat().st_size < 100:
                    sessions_to_fetch.append(session)
                    
            if not sessions_to_fetch:
                logger.info(f"⏭Skipping {year} Round {round_num}: {gp_name} (all {len(scheduled_sessions)} sessions already downloaded)")
                continue
                
            logger.info(f"Downloading {year} Round {round_num}: {gp_name} (fetching: {', '.join(sessions_to_fetch)})...")
            
            # calls the fetch_gp function from src.data_fetcher
            fetch_gp(year, round_num, sessions_to_fetch)
            
            # Short sleep between races to be polite to the API
            time.sleep(2)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted. Exiting...")
        sys.exit(0)
