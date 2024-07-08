from datasets import load_dataset
import argparse
import json
from pathlib import Path
import pyarrow.parquet as pq
import logging
import os
import random


def setup_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='/data/sapo/')
    parser.add_argument('--data', type=str, default='argilla/distilabel-capybara-dpo-7k-binarized')
    return parser.parse_args()


def load_and_process_data_deita_sft(dataset_name, split):
    try:
        dataset = load_dataset(dataset_name, split=split)
        reformatted_data = [{
            'real': message['messages']
        } for message in dataset]
        return reformatted_data
    except Exception as e:
        logging.error(f"Error loading or processing dataset: {e}")
        return []
        
def load_and_process_data_capybara_7k(dataset_name, split):
    try:
        dataset = load_dataset(dataset_name, split=split)
        reformatted_data = [{
            'real': message['chosen']
        } for message in dataset]
        return reformatted_data
    except Exception as e:
        logging.error(f"Error loading or processing dataset: {e}")
        return []

def save_to_json(data, path):
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        logging.error(f"Error saving data to {path}: {e}")


def main():
    args = setup_arg_parser()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    

    if args.data == 'argilla/distilabel-capybara-dpo-7k-binarized':
        train_data = load_and_process_data_capybara_7k(args.data, 'train')

        for section in train_data:
            for message in section['real']: 
                message['content'] = message['content'].lstrip()

        train_json_path = output_dir / 'train.json'
        save_to_json(train_data, train_json_path)

    if args.data == "HuggingFaceH4/deita-10k-v0-sft":
        train_data = load_and_process_data_deita_sft(args.data, 'train_sft')

        train_json_path = output_dir / 'train.json'

        for section in train_data:
            for message in section['real']: 
                message['content'] = message['content'].lstrip()
        
        save_to_json(train_data, train_json_path)

if __name__ == "__main__":
    main()