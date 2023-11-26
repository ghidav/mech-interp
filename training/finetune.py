import os
import sys
from typing import List
import fire
import torch
import transformers
from datasets import load_dataset
import logging
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ["WANDB_DISABLED"] = "true"

class Prompter(object):

    def __init__(self):
        self.template = {
            "prompt": "A chat between a user and an AI assistant. The assistant answers the user's questions.\n\n"
                      "### User: {instruction}\n### Assistant: ",
            "response_split": "### Assistant:",
        }

    def generate_prompt(
            self,
            instruction: str,
            output: str = None,

    ) -> str:
        res = self.template["prompt"].format(
            instruction=instruction
        )
        if output:
            res = f"{res}{output}"

        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


def train(
        base_model: str = "",  # the only required argument
        train_data_path: str = "",
        validation_data_path: str = "",
        output_dir: str = "./lora-alpaca",
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 4,
        learning_rate: float = 3e-4,
        cutoff_len: int = 300,
        lora_r: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules = None,
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        add_eos_token: bool = False,
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter()

    if lora_target_modules is None:
        lora_target_modules = [
            "q_proj",
            "v_proj",
        ]

    ##################
    # WILD DDP STUFF #
    ##################

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    ##################
    # Load Model     #
    ##################

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # fix llama tokenizer
    if "llama" in base_model.lower():
        logging.info(f"Setting missing PAD token to EOS token for {base_model}")
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast

        # tokenization is going to cutoff and we are going to manually add the eos token
        # if eos not there

        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token):

            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)


        # so this is little weird to me but...
        result["labels"] = result["input_ids"].copy()

        """
        in both LlamaForCausalLM and FalconForCausalLM, logits are shifted
        see also: https://github.com/tloen/alpaca-lora/issues/365
        
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        """

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)

        # we can use this to mask the instruction if we don't want to train on that
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                         user_prompt_len:]
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    ##################
    # Load Data      #
    ##################

    if train_data_path.endswith(".json") or train_data_path.endswith(".jsonl"):
        train_data = load_dataset("json", data_files=train_data_path)
    else:
        raise Exception("Training data must be in json or jsonl format.")

    if validation_data_path.endswith(".json") or validation_data_path.endswith(".jsonl"):
        validation_data = load_dataset("json", data_files=validation_data_path)
    else:
        raise Exception("Validation data must be in json or jsonl format.")

    train_data = train_data["train"].shuffle().map(generate_and_tokenize_prompt)
    validation_data = validation_data["train"].shuffle().map(generate_and_tokenize_prompt)

    ##################
    # Bless DDP      #
    ##################

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    ##################
    # Trainer        #
    ##################

    model.print_trainable_parameters()

    t_args = transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=10,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=50,
        save_steps=50,
        output_dir=output_dir,
        save_total_limit=30,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to=None,
        run_name=None)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=validation_data,
        args=t_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True))

    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
