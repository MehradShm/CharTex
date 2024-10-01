import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoModelForImageClassification, ViTModel
from transformers import AutoTokenizer, AutoModelForCausalLM, DonutProcessor
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import Trainer, TrainingArguments
from PIL import Image
import numpy as np
from torchvision import transforms
from datasets import load_dataset
from unids import DonutDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from peft import get_peft_model, LoraConfig, TaskType
from peft import prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from argparse import ArgumentParser
from lightning.pytorch.callbacks import ModelSummary
from pytorch_lightning.callbacks import ModelCheckpoint
import pickle
import multiprocessing
import os

class test_processor():
    def __init__(self, vision_processor, text_tokenizer):
        self.processor = vision_processor
        self.tokenizer = text_tokenizer

    def __call__(self, image):
        image_tensor = self.processor(image).unsqueeze(0)
        return image_tensor

class AlignmentMLP(nn.Module):
    def __init__(self, vision_dim, text_dim, hidden_dim):
        super(AlignmentMLP, self).__init__()
        self.fc1 = nn.Linear(vision_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, text_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class MultimodalModelPL(pl.LightningModule):
    def __init__(self, model, processor, train_dataset, val_dataset, config):
        super(MultimodalModelPL, self).__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config=config
        self.criterion = nn.CrossEntropyLoss()
        self.processor = processor

    def forward(self, pixel_values, input_ids):
        
        logits = self.model(pixel_values, input_ids)
        return logits

    def training_step(self, batch, batch_idx):
        pixel_values, input_ids, labels = batch['pixel_values'], batch['input_ids'], batch['labels']
        logits = self.forward(pixel_values, input_ids)
        logits = logits[:, 577:]  # logits shape: [4, 1024, 32000]
        logits = logits.contiguous().view(-1, logits.size(-1))  # Reshape to [4 * 1024, 32000]        
        labels = labels.contiguous().view(-1)  # Reshape to [4 * 1024]
        #loss = criterion(logits, labels)      
        loss = self.criterion(logits, labels)
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values, input_ids, labels = batch['pixel_values'], batch['input_ids'], batch['labels']
        logits = self.forward(pixel_values, input_ids)
        logits = logits[:, 577:]  # logits shape: [4, 1024, 32000]
        logits = logits.contiguous().view(-1, logits.size(-1))  # Reshape to [4 * 1024, 32000]        
        labels = labels.contiguous().view(-1)  # Reshape to [4 * 1024]
        loss = self.criterion(logits, labels)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['lr'])
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config["train_batch_sizes"], shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config["val_batch_sizes"], shuffle=False, num_workers=2)

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        save_path = os.path.join(self.config['result_path'], 'checkpoint-epoch='+str(self.current_epoch)+'-'+str(self.global_step))
        #self.model.save_pretrained(save_path)
        print(save_path, " | What the?")
        torch.save(self.model.alignment_mlp, save_path)
        #checkpoint['alignment'] = self.model.alignment_mlp



class MultimodalModel(nn.Module):
    def __init__(self, vision_model, alignment_mlp, text_model):
        super(MultimodalModel, self).__init__()
        self.vision_model = vision_model
        self.alignment_mlp = alignment_mlp
        self.text_model = text_model

    def forward(self, pixel_values, input_ids):

        tmp = self.vision_model(pixel_values)
        vision_outputs = tmp.last_hidden_state

        aligned_features = self.alignment_mlp(vision_outputs)

        text_embeddings = self.text_model.get_input_embeddings()(input_ids)
        combined_features = torch.cat((aligned_features, text_embeddings), dim=1)

        outputs = self.text_model(inputs_embeds=combined_features)

        return outputs.logits

def freeze_model_weights(model):
    for param in model.parameters():
        param.requires_grad = False

if __name__ == '__main__':
    base_dir = '/'.join(os.getcwd().split('/')[:3])

    vit_path = f'{base_dir}/work/models/vit'
    llama_path = f'{base_dir}/work/models/llama2'
    mistral_path = f'{base_dir}/work/models/mistral'
    gemma2_path = f'{base_dir}/work/models/gemma2'
    unichart_path = f'{base_dir}/work/models/unichart/Encoder'
    ds_path = f'{base_dir}/uniptds'
    gemma22_path = f'{base_dir}/work/models/gemma2-2B'
    
    image_folder = f'{base_dir}/content/tmp'

    multiprocessing.set_start_method('spawn')

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    vision_model = ViTModel.from_pretrained(vit_path)
    vision_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    alignment_mlp = AlignmentMLP(vision_dim=768, text_dim=4096, hidden_dim=1024)
    text_tokenizer = AutoTokenizer.from_pretrained(llama_path)
    text_model = LlamaForCausalLM.from_pretrained(llama_path)
    text_tokenizer.pad_token = text_tokenizer.eos_token
    text_tokenizer.padding_side = "right"
    
    processor = test_processor(vision_transform, text_tokenizer)
    
    multimodal_model = MultimodalModel(vision_model, alignment_mlp, text_model)
    
    freeze_model_weights(multimodal_model.vision_model)
    freeze_model_weights(multimodal_model.text_model)
    
    train_set = load_dataset(ds_path, split='train') 
    
    indices = []
    for i, sample in enumerate(train_set):
        if '<summarize_chart>' in sample['query'] or '<extract_data_table>'in sample['query']:
            indices.append(i)
    
    huh_tr = DonutDataset(ds_path, image_folder, 1024, processor, split = 'train', prompt_end_token = '<s_answer>', indices = indices[:128])
    huh_ev = DonutDataset(ds_path, image_folder, 1024, processor, prompt_end_token = '<s_answer>', indices = indices[128:144])
    
    config = {#"max_steps":args.max_steps,
                #"val_check_interval":0.2, # how many times we want to validate during an epoch
                "check_val_every_n_epoch":1,
                "log_every_n_steps":20,
                "gradient_clip_val":1,
                "num_training_samples_per_epoch": 128,
                "lr":0.001,
                "train_batch_sizes": 64,
                "val_batch_sizes": 1,
                "num_nodes": 1,
                "warmup_steps":5, # 800/8*30/10, 10%
                "result_path": os.path.join(base_dir, 'work/results'),
                "verbose": True,
              }
    
    model_module = MultimodalModelPL(multimodal_model, processor, huh_tr, huh_ev, config)
    
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(base_dir, 'results'), every_n_epochs=None, save_top_k=-1,  # Save all checkpoints
    every_n_train_steps=8)

    
    trainer = pl.Trainer(
            accelerator="gpu",
            max_epochs=2,
            devices=4,
            max_steps=10000000,
            check_val_every_n_epoch=1,
            log_every_n_steps=8,
            gradient_clip_val=1,
            num_nodes=1,
            precision="16-mixed", # we'll use mixed precision
            num_sanity_val_steps=0,
            #enable_checkpointing=True,
            default_root_dir=os.path.join(base_dir, 'results'),
            # logger=wandb_logger,
            callbacks=[checkpoint_callback],
      )
    
    # # Read the memory info from /proc/meminfo
    # with open('/proc/meminfo', 'r') as meminfo:
    #     lines = meminfo.readlines()

    # # Extract total and available memory in kB
    # mem_total = int([x for x in lines if 'MemTotal:' in x][0].split()[1])
    # mem_available = int([x for x in lines if 'MemAvailable:' in x][0].split()[1])

    # print(f"Total Memory: {mem_total / 1024:.2f} MB")
    # print(f"Available Memory: {mem_available / 1024:.2f} MB")
    
    trainer.fit(model_module)