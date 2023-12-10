import os
import inotify_simple
import threading
import logging
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Function to be executed when a new file is added
def on_new_file(event):
    print(f"New file added: {event.name}")

# Watcher thread that monitors the directory
def watch_directory(directory):
    try:
        fd = inotify_simple.INotify()
        wd = fd.add_watch(directory, inotify_simple.flags.CREATE)

        while True:
            print("inside loop")
            for event in fd.read():
                print("event found")
                if event.mask & inotify_simple.flags.CREATE:
                    on_new_file(event)
    except Exception as e:
        print("exception found")
        logger.exception("An error occurred in the watcher thread.")
    finally:
        try:
            print("trying to close")
            fd.close()
        except Exception as e:
            logger.exception("An error occurred while closing the INotify instance.")

# Example usage
if __name__ == "__main__":
    directory_to_watch = ""

    watcher_thread = threading.Thread(target=watch_directory, args=(directory_to_watch,))
    watcher_thread.start()

    try:
        # Your main program logic can run here
        while True:
           time.sleep(2)
           print("inside main thread") 
        # watcher_thread.join()
    except KeyboardInterrupt:
        watcher_thread.join()
