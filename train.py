import os
import argparse
import torch
import torch.distributed as dist
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_distributed():
    if "PBS_JOBID" in os.environ:
        local_rank = int(os.environ.get("PBS_VNODENUM", 0))
        world_size = int(os.environ.get("PBS_NUM_NODES", 1)) * int(
            os.environ.get("PBS_NUM_PPN", 8)
        )
        rank = int(os.environ.get("PBS_ARRAYID", 0))

        master_addr = os.environ.get("PBS_NODELIST", "localhost").split(",")[0]
        master_port = int(os.environ.get("PBS_JOBID", "12345")[-6:]) + 10000

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )

        torch.cuda.set_device(local_rank)
        return local_rank, world_size, rank
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--data_path", type=str, default="./data/train.jsonl")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--fsdp_config", type=str, default="full_shard auto_wrap")
    parser.add_argument("--use_flash_attention", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def get_model_config(model_name):
    model_configs = {
        "7b": {"torch_dtype": torch.bfloat16, "quantization": None},
        "13b": {"torch_dtype": torch.bfloat16, "quantization": None},
        "70b": {"torch_dtype": torch.bfloat16, "quantization": None},
        "llama-2-7b": {"torch_dtype": torch.bfloat16, "quantization": None},
        "llama-2-13b": {"torch_dtype": torch.bfloat16, "quantization": None},
        "llama-2-70b": {"torch_dtype": torch.bfloat16, "quantization": None},
    }

    for key in model_configs:
        if key in model_name.lower():
            return model_configs[key]
    return {"torch_dtype": torch.bfloat16, "quantization": None}


def load_model_and_tokenizer(args):
    logger.info(f"Loading model: {args.model_name}")

    model_config = get_model_config(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": model_config["torch_dtype"],
        "low_cpu_mem_usage": args.low_cpu_mem_usage,
    }

    if args.fsdp and args.model_name.lower().contains("70"):
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    return model, tokenizer


def load_datasets(args, tokenizer):
    logger.info(f"Loading dataset from: {args.data_path}")

    if args.data_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=args.data_path)
    elif args.data_path.endswith(".json"):
        dataset = load_dataset("json", data_files=args.data_path)
    elif args.data_path.endswith(".parquet"):
        dataset = load_dataset("parquet", data_files=args.data_path)
    else:
        dataset = load_dataset(args.data_path)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"] if "text" in examples else examples["content"],
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    return tokenized_dataset


def main():
    args = parse_args()
    set_seed(args.seed)

    local_rank, world_size, rank = setup_distributed()

    is_main = rank == 0

    if is_main:
        logger.info("Training configuration:")
        logger.info(f"  Model: {args.model_name}")
        logger.info(f"  Data: {args.data_path}")
        logger.info(f"  Output: {args.output_dir}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
        logger.info(f"  Learning rate: {args.learning_rate}")
        logger.info(f"  Epochs: {args.num_epochs}")
        logger.info(f"  FSDP enabled: {args.fsdp}")

    model, tokenizer = load_model_and_tokenizer(args)
    dataset = load_datasets(args, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        bf16=True,
        dataloader_num_workers=4,
        report_to="none",
        seed=args.seed,
        fsdp_sharding_strategy="full" if args.fsdp else None,
        fsdp_config={"auto_wrap": True, "backward_prefetch": "backward_prefetch"}
        if args.fsdp
        else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
    )

    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    if is_main:
        trainer.save_model(f"{args.output_dir}/final")
        tokenizer.save_pretrained(f"{args.output_dir}/final")
        logger.info("Training complete!")

    cleanup_distributed()


if __name__ == "__main__":
    main()
