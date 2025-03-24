"""
Utility functions for the XGBoost optimizer.
"""
import threading
import time
import config


def user_input_thread():
    """
    Background thread to handle user input for pausing/quitting the optimization.

    Listens for 'p' to pause/resume and 'q' to quit.
    """
    while True:
        user_input = input("Press 'p' to pause/resume or 'q' to quit: ")
        if user_input.lower() == 'p':
            config.should_pause = not config.should_pause
            print("Paused" if config.should_pause else "Resumed")
        elif user_input.lower() == 'q':
            print("Quitting...")
            config.should_pause = True
            break
        time.sleep(0.1)  # Small sleep to prevent high CPU usage


def start_user_input_thread():
    """
    Start the user input thread as a daemon thread.

    Returns:
        The started thread object
    """
    input_thread = threading.Thread(target=user_input_thread, daemon=True)
    input_thread.start()
    return input_thread