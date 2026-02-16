import subprocess
from pathlib import Path

try:
    # Run create_chat_set script from rulm
    module_directory = '/Users/user/Desktop/dev/ai/rulm/self_instruct'
    train_path = str(Path(module_directory) / 'data' / 'train.jsonl')
    val_path = str(Path(module_directory) / 'data' / 'val.jsonl')

    subprocess.run(
        [
            'python', '-m', 'src.data_processing.create_short_chat_set',
            train_path,
            val_path
        ],
        cwd=module_directory,
        check=True
    )
except ModuleNotFoundError as e:
    print(f"Error: {e}. Please ensure that all dependencies are installed.")

