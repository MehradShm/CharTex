from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
import json
import os

base_dir = '/'.join(os.getcwd().split('/')[:3])
dataset = load_from_disk("/home/msm97/scratch/datasets/Benchmarks/OpenCQA/hf_set")
print(dataset)

# # Load your JSON file
# with open(os.path.join(base_dir, 'data/ocqa.json'), 'r') as f:
#     data = json.load(f)

# # Create the DatasetDict with the three splits
# dataset_dict = DatasetDict({
#     'train': Dataset.from_list(data['train']),
#     'val': Dataset.from_list(data['val']),
#     'test': Dataset.from_list(data['test'])
# })

# dataset_dict.save_to_disk("/home/msm97/scratch/datasets/Benchmarks/OpenCQA/hf_set")