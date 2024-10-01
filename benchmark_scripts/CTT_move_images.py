import os
import shutil
import json

base_dir = "/".join(os.getcwd().split("/")[0:3])

def get_names(base_dir, subset, split):
    with open(f"{base_dir}/files/Chart-to-text-main/{subset}_dataset/dataset/dataset_split/{split}_index_mapping.csv", 'r') as f:
        split_names = f.readlines()
    
    split_names = split_names[1:]
    split_names = [x[:-1] for x in split_names]
    return split_names

train_names, val_names, test_names = get_names(base_dir, "pew", "train"), get_names(base_dir, "pew", "val"), get_names(base_dir, "pew", "test")
subset = "pew"
twocol_names, multicol_names = os.listdir(f"{base_dir}/files/Chart-to-text-main/{subset}_dataset/dataset/captions"), os.listdir(f"{base_dir}/files/Chart-to-text-main/{subset}_dataset/dataset/multiColumn/captions")
twocol_names, multicol_names = ['two_col-' + name for name in twocol_names], ['multi_col-' + name for name in multicol_names]
os.mkdir(f"{base_dir}/data/pew_images")

splits = [train_names, val_names, test_names]
for split_names in splits:
    for name in split_names:
        key = int('multi_col-' in name)
        address_map = {0: lambda x: f'{x}', 1: lambda x: f'multiColumn/{x}'}
        shutil.copy2(os.path.join(f"{base_dir}/files/Chart-to-text-main/{subset}_dataset/dataset/{address_map[key]('imgs')}",name.split('-')[1].split('.')[0]+'.png')
                    , f"{base_dir}/data/pew_images/{name.split('.')[0] + '.png'}")

subset = "statista"
train_names, val_names, test_names = get_names(base_dir, subset, "train"), get_names(base_dir, subset, "val"), get_names(base_dir, subset, "test")
twocol_names, multicol_names = os.listdir(f"{base_dir}/files/Chart-to-text-main/{subset}_dataset/dataset/captions"), os.listdir(f"{base_dir}/files/Chart-to-text-main/{subset}_dataset/dataset/multiColumn/captions")
twocol_names, multicol_names = ['two_col-' + name for name in twocol_names], ['multi_col-' + name for name in multicol_names]
os.mkdir(f"{base_dir}/data/statista_images")

splits = [train_names, val_names, test_names]
for split_names in splits:
    for name in split_names:
        key = int('multi_col-' in name)
        address_map = {0: lambda x: f'{x}', 1: lambda x: f'multiColumn/{x}'}
        shutil.copy2(os.path.join(f"{base_dir}/files/Chart-to-text-main/{subset}_dataset/dataset/{address_map[key]('imgs')}",name.split('-')[1].split('.')[0]+'.png')
                    , f"{base_dir}/data/statista_images/{name.split('.')[0] + '.png'}")