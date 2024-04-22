import os
import sys
import csv
import numpy as np

sys.path.append(os.path.abspath(os.pardir))
from utils import convert_csv_h5


def read_csv(file):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = np.array(list(reader))
        print(f"Read data shape from {file}: {data.shape}")
    return data

def join_data(data):
    joined_data = []
    for row in data:
        joined_data.append(' '.join(row))
    return joined_data

def write_to_file(data, file):
    with open(file, 'w') as f:
        for row in data:
            f.write(row + '\n')
    print(f"Data written to {file}")

def process_and_write_file(input_file, output_file, mode='a'):
    with open(input_file, 'r') as file_in:
        reader = csv.reader(file_in)
        with open(output_file, mode) as file_out:
            writer = csv.writer(file_out)
            for row in reader:
                writer.writerow(row)

def main():
    parent_dir = os.getcwd()  # Assuming current working directory is the base
    output_file = os.path.join(parent_dir, 'tuh_signals_4s.csv')

    # Ensure output file is empty or create it
    open(output_file, 'w').close()

    for i in range(7):
        file_path = os.path.join(parent_dir, f'signals_00{i+1}_128_4s.csv')
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist")
            continue

        # Process each file and append its data to the output file
        process_and_write_file(file_path, output_file)

if __name__ == "__main__":
    convert_csv_h5('tuh_signals_4s.csv')