import os
import argparse
import re
import shutil
import json


def copy_files_from_schedules(directory_path):
    # Iterate over subdirectories
    for root, dirs, files in os.walk(directory_path):
        print(dirs)
        if root.endswith("calculate_avg_wait_time") or root.endswith("calculate_twct"):
            if not any([f.endswith(".schdls") for f in files]):
                for file in files:
                    if file.endswith('.json'):
                        json_file_path = os.path.join(root, file)
                        with open(json_file_path, 'r') as json_file:
                            data = json.load(json_file)
                            schedules_path = data.get('schedules')
                            if schedules_path:
                                # Get the absolute path to the file to be copied
                                # Define a regular expression pattern to match the desired part of the path
                                pattern = r"\d{8}_\d{6}/.*/.*\.schdls"

                                # Use regex to find the matching part of the path
                                match = re.search(pattern, schedules_path)

                                if match:
                                    extracted_part = match.group()
                                    print("Extracted part of the path:", extracted_part)

                                    file_to_copy = os.path.join(directory_path, extracted_part)
                                    print(f"Copy from here: {file_to_copy}")

                                    # Copy the file to the current directory
                                    shutil.copy(file_to_copy, root)
                                    print(f"Copied {file_to_copy} to {root}")
                                else:
                                    print("No match found.")

                                # Copy the file to the current directory
                                # shutil.copy(file_to_copy, root)
                                # print(f"Copied {file_to_copy} to {root}")


if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Copy schedules based on path from JSON files')
    parser.add_argument('directory', type=str, help='Path to the directory')
    args = parser.parse_args()

    # Check if the provided directory exists
    if not os.path.exists(args.directory):
        print(f"The directory {args.directory} does not exist.")
        exit()

    # Call the function to copy files based on schedules
    copy_files_from_schedules(args.directory)
