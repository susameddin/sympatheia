from datasets import load_dataset
import datetime
import os
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from transformers import AutoModel, AutoTokenizer
import yaml

os.environ["WANDB_PROJECT"] = "Sympatheia-11emo-34K"

# Reuse the same hyperparameter config as train.py
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

run_name = f"sympatheia-11emo-34k-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
output_dir = f"./experiments/{run_name}"
os.makedirs(output_dir, exist_ok=True)

# Persist run config for reproducibility
with open(os.path.join(output_dir, "config.yaml"), "w") as f:
    yaml.dump(config, f)

MAX_LENGTH = config["max_length"]

# VA format dataset (valence/arousal values in system prompt text)
# data_files = {
#     "train": "../../../Datasets/OpenS2S_Qwen3TTS/glm4voice_va_format/train.jsonl",
#     "validation": "../../../Datasets/OpenS2S_Qwen3TTS/glm4voice_va_format/eval.jsonl",
# }
data_files = {
    "train": "/engram/naplab/users/sd3705/Datasets/Sympatheia_11Emo_Combined/train.jsonl",
    "validation": "/engram/naplab/users/sd3705/Datasets/Sympatheia_11Emo_Combined/eval.jsonl",
}
raw_datasets = load_dataset("json", data_files=data_files)
train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["validation"]
print("VA format dataset loaded")

peft_config = LoraConfig(
    r=config["lora_r"],
    lora_alpha=config["lora_alpha"],
    target_modules=config["lora_trainable"],
    lora_dropout=config["lora_dropout"],
    bias="none",
    task_type="CAUSAL_LM",
)

sft_config = SFTConfig(
    output_dir=output_dir,
    max_seq_length=MAX_LENGTH,
    learning_rate=config["learning_rate"],
    per_device_train_batch_size=config["train_bsz"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    per_device_eval_batch_size=config["eval_bsz"],
    warmup_steps=config["warmup_steps"],
    logging_strategy="steps",
    logging_steps=config["logging_steps"],
    eval_strategy="steps",
    eval_steps=config["save_steps"],
    save_strategy="steps",
    save_steps=config["save_steps"],
    num_train_epochs=config["num_epochs"],
    bf16=True,
    gradient_checkpointing=False,
    deepspeed="ds_config.json",
    dataset_text_field="text",
    report_to="wandb",
    run_name = f"glm-model-opens2s-qwen3tts-va-text-lora-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
)

model = AutoModel.from_pretrained("THUDM/glm-4-voice-9b", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-voice-9b", trust_remote_code=True)

trainer = SFTTrainer(
    model,
    tokenizer=tokenizer,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
)

trainer.train()
