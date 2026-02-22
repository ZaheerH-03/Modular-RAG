import os
from llama_index.core import SimpleDirectoryReader

base_path = "E:/Documents/data_Science/ModularRAG/data"

def get_data_loader(year: int, branch: str) -> SimpleDirectoryReader:
    # 1. Create a LIST of directories, not a concatenated string
    dirs_to_load = []
    if year >= 1: dirs_to_load.append(f"{base_path}/1st_yr_{branch}")
    if year >= 2: dirs_to_load.append(f"{base_path}/2nd_yr_{branch}")
    if year >= 3: dirs_to_load.append(f"{base_path}/3rd_yr_{branch}")
    if year >= 4: dirs_to_load.append(f"{base_path}/4th_yr_{branch}")

    # 2. Verify directories exist and collect all files
    all_files = []
    for d in dirs_to_load:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for file in files:
                    all_files.append(os.path.join(root, file))
        else:
            print(f"⚠️ Warning: Directory not found: {d}")

    # 3. Pass the list of FILES to the reader
    if not all_files:
        raise ValueError(f"No files found for Year {year}, Branch {branch}!")

    return SimpleDirectoryReader(input_files=all_files)