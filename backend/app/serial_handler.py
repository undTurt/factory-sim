import serial # type: ignore
import json
import logging
from threading import Thread, Event
from queue import Queue
import time
import os
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add this near the top of the file, after imports
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'arduino_data')
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ArduinoSerialHandler:
    def __init__(self, port='COM3', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.is_running = Event()
        self.data_queue = Queue()
        self.latest_data = {}

    def connect(self):
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1
            )
            time.sleep(2)  # Wait for Arduino to reset
            logger.info(f"Connected to Arduino on {self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Arduino: {str(e)}")
            return False

    def start_monitoring(self):
        if not self.serial or not self.serial.is_open:
            if not self.connect():
                return False

        self.is_running.set()
        Thread(target=self._read_serial, daemon=True).start()
        logger.info("Started Arduino monitoring")
        return True

    def stop_monitoring(self):
        self.is_running.clear()
        if self.serial and self.serial.is_open:
            self.serial.close()
        logger.info("Stopped Arduino monitoring")

        # Modify the _read_serial method in ArduinoSerialHandler class to write JSON files:
        def _read_serial(self):
            while self.is_running.is_set():
                if self.serial.in_waiting:
                    try:
                        line = self.serial.readline().decode('utf-8').strip()
                        if line:
                            try:
                                # Parse the pipe-separated data
                                values = line.split('|')
                                if len(values) == 7:  # Ensure all values are present
                                    data = {
                                        "count": int(values[0].strip()),
                                        "time_between_ms": float(values[1].strip()),
                                        "count_per_min": float(values[2].strip()),
                                        "avg_count_per_min": float(values[3].strip()),
                                        "total_time_s": float(values[4].strip()),
                                        "energy_used_kwh": float(values[5].strip()),
                                        "total_energy_kwh": float(values[6].strip()),
                                        "timestamp": datetime.now().isoformat()
                                    }
                                    
                                    self.latest_data = data
                                    self.data_queue.put(data)
                                    
                                    # Write to JSON file
                                    filename = f"manufacturing_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                                    filepath = os.path.join(DATA_DIR, filename)
                                    
                                    with open(filepath, 'w') as f:
                                        json.dump(data, f, indent=2)
                                        
                                    logger.debug(f"Saved manufacturing data: {data}")
                                    
                            except (ValueError, IndexError) as e:
                                logger.warning(f"Invalid data format: {line}, Error: {str(e)}")
                                
                    except Exception as e:
                        logger.error(f"Error reading serial data: {str(e)}")
    def get_latest_data(self):
        return self.latest_data

    def send_command(self, command):
        try:
            if self.serial and self.serial.is_open:
                self.serial.write(f"{command}\n".encode())
                logger.debug(f"Sent command: {command}")
                return True
        except Exception as e:
            logger.error(f"Failed to send command: {str(e)}")
        return False

class ArduinoDataHandler(FileSystemEventHandler):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.latest_data = {}
        
    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith('.json'):
            return
            
        try:
            with open(event.src_path, 'r') as f:
                data = json.load(f)
                self.latest_data = data
                logger.info(f"Processed new data file: {event.src_path}")
                
            # Clean up old file after processing
            os.remove(event.src_path)
        except Exception as e:
            logger.error(f"Error processing data file: {str(e)}")

class ArduinoDataMonitor:
    def __init__(self, data_dir='arduino_data'):
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', data_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.handler = ArduinoDataHandler(self.data_dir)
        self.observer = Observer()
        self.observer.schedule(self.handler, self.data_dir, recursive=False)
        
    def start_monitoring(self):
        self.observer.start()
        logger.info(f"Started monitoring Arduino data directory: {self.data_dir}")
        return True
        
    def stop_monitoring(self):
        self.observer.stop()
        self.observer.join()
        logger.info("Stopped monitoring Arduino data directory")
        
    def get_latest_data(self):
        return self.handler.latest_data