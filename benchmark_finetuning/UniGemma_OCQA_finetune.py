import torch
from torch import nn
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, DonutProcessor
from transformers import DonutProcessor, VisionEncoderDecoderModel, DonutSwinModel
from torchvision import transforms
from datasets import load_dataset
#from FThfDataModel import DonutDataset
from FTJsonDataModel import DonutDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from peft import get_peft_model, LoraConfig, TaskType
from peft import prepare_model_for_kbit_training
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from lightning.pytorch.callbacks import ModelSummary
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import argparse

base_dir = '/'.join(os.getcwd().split('/')[:3])

parser = argparse.ArgumentParser(description='Train Chart Transformer')
#/home/msm97/scratch/datasets/fixed_format_instructions.json
parser.add_argument('--benchmark-path', type=str, default = f"{base_dir}/data/ocqa.json", help='Path to the input json file')
#parser.add_argument('--benchmark-path', type=str, default = f"/home/msm97/scratch/datasets/fixed_format_instructions.json", help='Path to the input json file')
parser.add_argument('--output-path', type=str, default = "/home/msm97/scratch/model_checkpoints/OCQA/1", help='Path to the output folder')
parser.add_argument('--images-path', type=str, default=f'{base_dir}/data/chart_images', help='Path to the images Folder')
parser.add_argument('--model-path', type=str, default = None, help='Path to the model checkpoint')
parser.add_argument('--gemma22-path', type=str, default = f'{base_dir}/work/models/gemma2-2B', help='Path to the model checkpoint')
parser.add_argument('--unichart-path', type=str, default = f'{base_dir}/work/models/unichart/Encoder', help='Path to the model checkpoint')
parser.add_argument('--learning-rate', type=float, default=0.00002, help='Learning rate Value')
parser.add_argument('--context-length', type=int, default=768, help='Context Length Value')
parser.add_argument('--checkpoint-steps', type=int, default=1352, help='Steps needed to record a checkpoint')
parser.add_argument('--epochs', type=int, default=10, help='Total Epochs')
parser.add_argument('--train-batchsize', type=int, default=1, help='size of training batches')

args = parser.parse_args()
                    


unichart_path = f'{base_dir}/work/models/unichart/Encoder'
gemma22_path = f'{base_dir}/work/models/gemma2-2B'

# for OCQA
# benchmark_path = f'{base_dir}/data/ocqa.json'

# for ChartFC

#for C2T
#benchmark_path = f'{base_dir}/data/statista_data.json' 
#for CQA
benchmark_path = f'{base_dir}/CQABench' 

#model_path = "/home/msm97/scratch/model_checkpoints/ins_tuning/model-checkpoint-epoch=2-104000"

#for OCQA
#image_folder = f'{base_dir}/data/chart_images'
#for ChartFC
#image_folder = f'{base_dir}/data/charts_seaborn_v5'
#for C2T
#image_folder = f'{base_dir}/data/statista_images'


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class unichart_processor():
    def __init__(self, vision_processor, text_tokenizer):
        self.processor = vision_processor
        self.tokenizer = text_tokenizer

    def __call__(self, image):
        image_tensor = self.processor(image, return_tensors="pt")
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
        logits = logits[:,900:]
        logits = logits.contiguous().view(-1, logits.size(-1))
        labels = labels.contiguous().view(-1)
        #loss = criterion(logits, labels)      
        loss = self.criterion(logits, labels)
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values, input_ids, labels = batch['pixel_values'], batch['input_ids'], batch['labels']
        logits = self.forward(pixel_values, input_ids)
        logits = logits[:,900:]
        logits = logits.contiguous().view(-1, logits.size(-1))
        labels = labels.contiguous().view(-1)
        loss = self.criterion(logits, labels)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['lr'])
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.config["train_batch_sizes"], shuffle=True, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.config["val_batch_sizes"], shuffle=False, num_workers=1)

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        model_path = os.path.join(self.config['result_path'], 'model-checkpoint-epoch='+str(self.current_epoch)+'-'+str(self.global_step))
        torch.save(self.model,model_path)


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

uni_processor = DonutProcessor.from_pretrained(args.unichart_path)
image_processor = uni_processor.image_processor
text_tokenizer = AutoTokenizer.from_pretrained(gemma22_path)
text_tokenizer.pad_token = text_tokenizer.eos_token
text_tokenizer.padding_side = "right"
processor = unichart_processor(image_processor, text_tokenizer)

if args.model_path:
    multimodal_model = torch.load(args.model_path)

else:
    vision_model = DonutSwinModel.from_pretrained(args.unichart_path)
    alignment_mlp = AlignmentMLP(vision_dim=1024, text_dim=2304, hidden_dim=4096)
    text_tokenizer = AutoTokenizer.from_pretrained(args.gemma22_path)
    text_model = AutoModelForCausalLM.from_pretrained(args.gemma22_path)
    multimodal_model = MultimodalModel(vision_model, alignment_mlp, text_model)

def freeze_model_weights(model):
    for param in model.parameters():
        param.requires_grad = False

freeze_model_weights(multimodal_model.vision_model)

huh_tr = DonutDataset(args.benchmark_path, args.images_path, args.context_length, processor, split = 'train', prompt_end_token = '</ins>')#, indices = list(range(24)))
huh_ev = DonutDataset(args.benchmark_path, args.images_path, args.context_length, processor, split = 'test', prompt_end_token = '</ins>')#, indices = list(range(4)))

print(len(huh_tr), " @@@ ", len(huh_ev))


config = {#"max_steps":args.max_steps,
                #"val_check_interval":0.2, # how many times we want to validate during an epoch
                "check_val_every_n_epoch":100,
                "log_every_n_steps":2,
                "gradient_clip_val":1,
                #"num_training_samples_per_epoch": 128,
                "lr":args.learning_rate,
                "train_batch_sizes": args.train_batchsize,
                "val_batch_sizes": 1,
                "num_nodes": 1,
                "warmup_steps":5, # 800/8*30/10, 10%
                "result_path": args.output_path,
                "verbose": True,
              }
    
model_module = MultimodalModelPL(multimodal_model, processor, huh_tr, huh_ev, config)

checkpoint_callback = ModelCheckpoint(dirpath= args.output_path, every_n_epochs=None, save_top_k=-1,  # Save all checkpoints
every_n_train_steps=args.checkpoint_steps)


trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=args.epochs,
        devices=4,
        max_steps=1000000,
        check_val_every_n_epoch=100,
        log_every_n_steps=8,
        gradient_clip_val=1,
        num_nodes=1,
        precision="16-mixed", # we'll use mixed precision
        num_sanity_val_steps=0,
        #enable_checkpointing=True,
        #default_root_dir=os.path.join(base_dir, 'results'),
        # logger=wandb_logger,
        callbacks=[checkpoint_callback],
  )


trainer.fit(model_module)