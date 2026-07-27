"""Stage 5 extra credit — QLoRA fine-tune a small local model on the SFT seed
data produced by build_answer_sft_data.py, to improve JSON validity/schema
adherence without a full fine-tune.

Loss is applied only to the assistant (target JSON) tokens: the prompt is
masked out (labels = -100) so the model isn't penalized for the fixed
instruction/schema text, only for reproducing the correct structured answer.

Usage:
    uv run python src/train_answer_lora.py \
        --train_jsonl data/answer_sft_seed.local_qwen3b.jsonl \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --out_dir adapters/stage5-qwen25-15b-json-lora \
        --num_train_epochs 3 \
        --learning_rate 2e-4
"""

import argparse
import json
import os


def load_dataset(path, tokenizer, max_length):
    import torch

    examples = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            messages = rec["messages"]
            prompt_messages = messages[:-1]
            full_text = tokenizer.apply_chat_template(messages, tokenize=False)
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True,
            )

            full_ids = tokenizer(full_text, truncation=True, max_length=max_length,
                                  return_tensors=None)["input_ids"]
            prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length,
                                    return_tensors=None)["input_ids"]
            prompt_len = min(len(prompt_ids), len(full_ids))

            labels = list(full_ids)
            for i in range(prompt_len):
                labels[i] = -100

            examples.append({
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            })
    return examples


class SFTCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        import torch
        max_len = max(len(ex["input_ids"]) for ex in batch)
        pad_id = self.tokenizer.pad_token_id

        input_ids, attention_mask, labels = [], [], []
        for ex in batch:
            pad = max_len - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [pad_id] * pad)
            attention_mask.append(ex["attention_mask"] + [0] * pad)
            labels.append(ex["labels"] + [-100] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser(description="Stage 5 extra credit: QLoRA SFT")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out_dir", default="adapters/stage5-lora")
    ap.add_argument("--num_train_epochs", type=float, default=3)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--max_length", type=int, default=1536)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        Trainer, TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb_config, device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_examples = load_dataset(args.train_jsonl, tokenizer, args.max_length)
    print(f"Loaded {len(train_examples)} SFT examples from {args.train_jsonl}")

    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        bf16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ListDataset(train_examples),
        data_collator=SFTCollator(tokenizer),
    )
    trainer.train()

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved LoRA adapter to {args.out_dir}")


if __name__ == "__main__":
    main()
