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
import os


device = "cuda:0"

vit_path = '/home/msm97/scratch/models/models--google--vit-base-patch16-384/snapshots/2960116e809e2fca84146dbb240289aee7db4827'

llama_path = '/home/msm97/scratch/models/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6'

ds_path = '/home/msm97/scratch/datasets/Unichart/uniptds'
base_dir = '/'.join(os.getcwd().split('/')[:3])
image_folder = f'{base_dir}/content/tmp'


class test_processor():
    def __init__(self, vision_processor, text_tokenizer):
        self.processor = vision_processor
        self.tokenizer = text_tokenizer

    def __call__(self, image):
        image_tensor = vision_transform(image).unsqueeze(0)
        return image_tensor

vision_model = ViTModel.from_pretrained(vit_path)
vision_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


bnb_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )

text_tokenizer = AutoTokenizer.from_pretrained(llama_path)
text_model = LlamaForCausalLM.from_pretrained(llama_path, quantization_config = bnb_config) #device_map = device1) 

text_model.gradient_checkpointing_enable()
text_model = prepare_model_for_kbit_training(text_model)


# TO_DO: Fix the target modules, different for Attention
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Targeted modules
    lora_dropout=0.1,  # Dropout rate
    bias="none",  # Bias setting
    task_type=TaskType.CAUSAL_LM,  # Task type
)

text_model = get_peft_model(text_model, lora_config)


special_tokens = text_tokenizer.special_tokens_map
text_tokenizer.pad_token = text_tokenizer.eos_token
text_tokenizer.padding_side = "right"


processor = test_processor(vision_transform, text_tokenizer)

huh_tr = DonutDataset(ds_path, image_folder, 1024, processor, split = 'train', prompt_end_token = '<s_answer>', sw = 0, ew = 104)
huh_ev = DonutDataset(ds_path, image_folder, 1024, processor, prompt_end_token = '<s_answer>', sw = 104, ew = 120)

dataloader = DataLoader(huh_tr, batch_size=1, shuffle=True, num_workers=4)

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

alignment_mlp = AlignmentMLP(vision_dim=768, text_dim=4096, hidden_dim=1024)

class MultimodalModel(nn.Module):
    def __init__(self, vision_model, alignment_mlp, text_model):
        super(MultimodalModel, self).__init__()
        self.vision_model = vision_model
        self.alignment_mlp = alignment_mlp
        self.text_model = text_model

    def forward(self, text_input_ids, image_input,
                # labels
                ):

        tmp = self.vision_model(image_input)
        vision_outputs = tmp.last_hidden_state[0]

        aligned_features = self.alignment_mlp(vision_outputs)

        text_embeddings = self.text_model.get_input_embeddings()(text_input_ids)
        combined_features = torch.cat((aligned_features.unsqueeze(0), text_embeddings), dim=1)

        outputs = self.text_model(inputs_embeds=combined_features)

        # For Trainer
        # if labels is not None:
        #     loss_fn = nn.CrossEntropyLoss()
        #     logits = outputs.logits[:,577:]
        #     logits = logits.view(-1, 128256)  # Shape: [1024, 128256]
        #     labels = labels.view(-1)
        #     loss = loss_fn(logits, labels)
        # return {"loss": outputs.loss, "logits": outputs.logits}

        #For Loop
        return {"logits": outputs.logits}

# Instantiate the custom multimodal model
multimodal_model = MultimodalModel(vision_model, alignment_mlp, text_model)

multimodal_model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(multimodal_model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

traindl, evaldl = DataLoader(huh_tr, batch_size=1, shuffle=True), DataLoader(huh_ev, batch_size=1, shuffle=True)


# TRAAAAAAAAAAAAAAAAAAAINEEEEEEEEER


# training_args = TrainingArguments(
#     output_dir="./results",
#     num_train_epochs=3,
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir="./logs",
#     logging_steps=10,
#     evaluation_strategy="epoch",
#     # save_strategy="epoch",
#     # load_best_model_at_end=True,
#     # save_total_limit=3,
#     # # Use multiple GPUs
#     dataloader_num_workers=1,
#     fp16=True,  # Enable 16-bit precision training if applicable
#     # report_to="none",
#     gradient_accumulation_steps=8,
# )
# for i in range(torch.cuda.device_count()):
#     print(f"GPU {i}:")
#     print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
#     print(f"  Reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
#     print(f"  Total: {torch.cuda.get_device_properties(i).total_memory / 1024**2:.2f} MB")

# trainer = Trainer(
#     model=multimodal_model,
#     args=training_args,
#     train_dataset=huh_tr,
#     #eval_dataset=huh_ev,
# )

# # Start training
# trainer.train()

# exit()

# TRAAAAAAAAAAAAAAAAAAAAAAAAAAAIN



for epoch in range(num_epochs):
    # Training phase
    multimodal_model.train()
    running_loss = 0.0

    for batch in dataloader:
        text_ids, image_input, labels = torch.Tensor(batch["image_input"]).to(device), torch.Tensor(batch["text_input_ids"].to(device)), torch.Tensor(batch["label_ids"].to(device))  
        optimizer.zero_grad()
        outputs = multimodal_model(text_ids, image_input)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    train_loss = running_loss / len(dataloader)


multimodal_model.eval()
val_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in evaldl:
        inputs, labels = inputs.to(device), labels.to(device)  # Send data to GPU
        
        outputs = multimodal_model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        
        # Get the predicted class
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_loss = val_loss / len(val_dataloader)
val_accuracy = correct / total

print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

print("Training Finished!")




image_tensor = vision_transform(image).unsqueeze(0)

outputs = multimodal_model(text_input_ids, image_tensor)
logits = outputs.logits
predicted_token_ids = logits.argmax(dim=-1)


tmp = text_model(text_input_ids)
logits = tmp.logits
predicted_token_ids = logits.argmax(dim=-1)
decoded_text = text_tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)

