import subprocess
import sys

def execute_script(script_path):
    """
    Executes a Python script and waits for it to finish.

    Parameters:
        script_path (str): Path to the Python script to execute.
    """
    try:
        result = subprocess.run([sys.executable, script_path], check=True)
        print(f"Execution of {script_path} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing {script_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Execute preprocess_paper_detection.py
    preprocess_script = "c:\\Users\\Owner\\Desktop\\preprocessing\\preprocessing\\preprocess_paper_detection.py"
    execute_script(preprocess_script)

    # Execute train_efficientnet.py
    train_script = "c:\\Users\\Owner\\Desktop\\preprocessing\\training\\train_efficientnet.py"
    execute_script(train_script)