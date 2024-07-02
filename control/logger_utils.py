import os
import sys
from datetime import datetime
import subprocess

class Tee:
    def __init__(self, name, mode):
        """
        The Tee class captures output and redirects it both to a file and the terminal.

        Parameters:
        - name (str): The name of the file where the output will be saved.
        - mode (str): The mode for opening the file (e.g., 'a' for appending to the file).
        """
        log_dir = os.path.dirname(name)
        os.makedirs(log_dir, exist_ok=True)
        self.name = name
        self.mode = mode
        self.stdout = sys.stdout

    def write(self, data):
        """
        Writes data both to the file and the terminal.

        Parameters:
        - data (str): The data to write.
        """
        with open(self.name, self.mode, encoding='utf-8') as f:
            f.write(data)
        self.stdout.write(data)

    def flush(self):
        """
        Flushes buffered data to the file and terminal.
        """
        self.stdout.flush()

class Logger:
    def __init__(self, log_file):
        """
        The Logger class manages logging messages to a file and displaying them in the terminal.

        Parameters:
        - log_file (str): The path to the log file.
        """
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        try:
            if os.path.exists(log_file):
                os.remove(log_file)
        except PermissionError as e:
            print(f"PermissionError: {e}. Make sure the file is not being used by another process.")
        sys.stdout = Tee(log_file, 'a')

    def log(self, message):
        """
        Writes a message to the log file with date and time.

        Parameters:
        - message (str): The message to write to the log file.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f'{timestamp} - {message}\n'
        sys.stdout.write(full_message)

    def run_script(self, script_name, args=None, python_executable=sys.executable):
        """
        Runs a script and redirects its output to the log file.

        Parameters:
        - script_name (str): The name of the script to run.
        - args (list): A list of arguments to pass to the script.
        - python_executable (str): The path to the Python interpreter.
        """
        command = [python_executable, script_name] + (args if args else [])
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        sys.stdout.write(stdout)
        sys.stderr.write(stderr)
        if process.returncode == 0:
            self.log(f"{script_name} completed successfully.")
        else:
            self.log(f"{script_name} failed with error: {stderr}")
