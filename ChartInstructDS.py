import json, os
import random
from typing import Any, List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import DonutProcessor
from datasets import load_dataset, load_from_disk
import json
from instructions import FACT, OCQA, CQA, CODE, COT, SUMMARY
import random

added_tokens = []

class DonutDataset(Dataset):
    """
    DonutDataset which is saved in huggingface datasets format. (see details in https://huggingface.co/docs/datasets)
    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
    and it will be converted into input_tensor(vectorized image) and input_ids(tokenized string).
    Args:
        dataset_name_or_path: name of dataset (available at huggingface.co/datasets) or the path containing image files and metadata.jsonl
        max_length: the max number of tokens for the target sequences
        split: whether to load "train", "validation" or "test" split
        ignore_id: ignore_index for torch.nn.CrossEntropyLoss
        task_start_token: the special token to be fed to the decoder to conduct the target task
        prompt_end_token: the special token at the end of the sequences
        sort_json_key: whether or not to sort the JSON keys
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        images_folder: str,
        max_length: int,
        processor : DonutProcessor = None,
        split: str = "train",
        ignore_id: int = -100,
#        task_start_token: str = "<s>",
        prompt_end_token: str = None,
        sort_json_key: bool = True,
        indices: List = None
    ):
        super().__init__()

        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.prompt_end_token = prompt_end_token #if prompt_end_token else task_start_token
        self.sort_json_key = sort_json_key
        self.images_folder = images_folder

        # dataset_dict = load_from_disk(dataset_name_or_path)
        # self.dataset = dataset_dict.select(indices)
        with open(dataset_name_or_path, 'r') as file:
            all_data = [json.loads(line) for line in file]
        # tasktype, imgname, query, label
        self.task2ins = {'REASONING':COT, 'FACT':FACT, 'OCQA':OCQA, 'CODE':CODE, 'NOVEL':OCQA, 'SUMMARY':SUMMARY}

        if indices:
            self.dataset = [all_data[0][i] for i in indices]
        else:
            self.dataset = all_data[0]
        self.dataset_length = len(self.dataset)

        self.processor = processor

        self.prompt_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)

    
    def __len__(self) -> int:
        return self.dataset_length - 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels
        Convert gt data into input_ids (tokenized string)
        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        sample = self.dataset[idx]

        # input_tensor
        img_path = os.path.join(self.images_folder, sample['imgname'])
        img = Image.open(img_path)

        #for VIT
        # pixel_values = self.processor(img.convert("RGB"))
        # input_tensor = pixel_values.squeeze()

        #for Unichart
        pixel_values = self.processor(img.convert("RGB")).pixel_values
        input_tensor = pixel_values.squeeze()

        random_int = random.randint(0, 5)

        #processed_parse = self.task2ins[sample['task_type']][random_int] + "\n" + sample['query'] + " " + self.prompt_end_token + " " + sample['label'] + self.processor.tokenizer.eos_token
        if sample['task_type'] == 'REASONING':
            processed_parse = "<ins> " + str(sample['query']).split("\n")[-1].split('"')[-1] + self.prompt_end_token+ "\n" + str(sample['label']) + self.processor.tokenizer.eos_token
        else:
            processed_parse = "<ins> " + str(sample['query']).split("\n")[-1] + self.prompt_end_token+ "\n" + str(sample['label']) + self.processor.tokenizer.eos_token
        input_ids = self.processor.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == self.processor.tokenizer.pad_token_id
            ] = self.ignore_id  # model doesn't need to predict pad token
           # print(labels.shape, " ^^^^ ", labels)
            labels[
                : torch.nonzero(torch.Tensor([label == self.prompt_end_token_id for label in labels])).sum() + 1
            ] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
            #return input_tensor, input_ids, labels
            tmp = {"pixel_values": input_tensor.to(torch.bfloat16), "input_ids": input_ids, "labels": labels}
            #tmp = (input_tensor,input_ids,labels)
            #print(len(tmp), " ** HUH **")
            return tmp
        else:
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()  # return prompt end index instead of target output labels
            return {"image_input": input_tensor, "text_input_ids": input_ids, "label":prompt_end_index, "label_ids": processed_parse}