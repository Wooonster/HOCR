import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import swanlab

from transformers import (
    AutoTokenizer, DataCollatorForSeq2Seq,
    Qwen2_5_VLForConditionalGeneration, AutoProcessor, 
    Trainer, TrainingArguments
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from qwen_vl_utils import process_vision_info

from swanlab.integration.transformers import SwanLabCallback


# model_id = 'Qwen/Qwen2.5-VL-7B-Instruct-AWQ'
student_save_dir = './vl3b/'
teacher_save_dir = './vl7b/'

class MyTrainer(Trainer):
    def __init__(
        self,
        model=None,
        teacher_model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=None,
        preprocess_logits_for_metrics=None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.teacher_model = teacher_model
        
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     outputs = model(**inputs)
    #     with torch.no_grad():
    #         teacher_outputs = self.teacher_model(**inputs)

    #     loss, logits = outputs.loss, outputs.logits
    #     teacher_logits = teacher_outputs.logits

    #     if logits.shape[-1] != teacher_logits.shape[-1]:
    #         # # padding
    #         # gap = teacher_logits.shape[-1] - logits.shape[-1]
    #         # if gap > 0:
    #         #     pad = torch.zeros((logits.shape[0], logits.shape[1], gap)).to(logits.device)
    #         #     logits = torch.cat([logits, pad], dim=-1)

    #         # truncation
    #         teacher_logits = teacher_logits[:, :, :logits.shape[-1]]

    #     labels = inputs['labels']
    #     fkl = forward_kl_diver(logits, teacher_logits, labels, padding_id=-100, temperature=2.0)

    #     return (fkl, outputs) if return_outputs else fkl

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)

        loss, logits = outputs.loss, outputs.logits
        teacher_logits = teacher_outputs.logits

        if logits.shape[-1] != teacher_logits.shape[-1]:
            # # padding
            # gap = teacher_logits.shape[-1] - logits.shape[-1]
            # if gap > 0:
            #     pad = torch.zeros((logits.shape[0], logits.shape[1], gap)).to(logits.device)
            #     logits = torch.cat([logits, pad], dim=-1)

            # truncation
            teacher_logits = teacher_logits[:, :, :logits.shape[-1]]

        labels = inputs['labels']
        fkl = forward_kl_diver(logits, teacher_logits, labels, padding_id=-100, temperature=2.0)

        return (fkl, outputs) if return_outputs else fkl


def forward_kl_diver(
    logits,
    teacher_logits,
    target,
    padding_id,
    reduction='sum',
    temperature=1.0,
):
    """
    q = argmin_q int_x p log(p/q) dx
    p -> teacher model
    q -> student model
    """
    logits = logits / temperature  # student model: q(x)
    teacher_logits = teacher_logits / temperature  # teacher model: p(x)
    
    log_probs = torch.log_softmax(logits, -1, dtype=torch.float32)  # log q(x)
    teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)  # p(x)
    teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)  # log p(x)

    forward_kl = teacher_probs * (teacher_log_probs - log_probs)
    forward_kl = forward_kl.sum(-1)  # 积分

    if reduction == 'sum':
        pad_mask = target.eq(padding_id)
        forward_kl = forward_kl.masked_fill_(pad_mask, 0.0)
        forward_kl = forward_kl.sum()

    return forward_kl



if __name__ == '__main__':
    # Lora Config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,  # 训练模式
        r=64,  # Lora 秩
        lora_alpha=16,
        lora_dropout=0.05,  # Dropout 比例
        bias="none",
    )

    teacher_lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=True,  # 推理模式
        r=64,  # Lora 秩
        lora_alpha=16,
        lora_dropout=0.05,  # Dropout 比例
        bias="none",
    )

    # student model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        student_save_dir, torch_dtype=torch.float16, 
        attn_implementation="flash_attention_2", device_map="auto"
    )
    model = get_peft_model(model, lora_config)
    model.cuda()
    print('student model params')
    print(model.print_trainable_parameters())

    # teacher model
    teacher_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        teacher_save_dir, torch_dtype=torch.float16, 
        attn_implementation="flash_attention_2", device_map="auto"
    )
    teacher_model = PeftModel.from_pretrained(
        teacher_model, model_id="./output/Qwen2-VL-7B/checkpoint-134/", 
        config=teacher_lora_config
    )
    teacher_model.cuda()
    teacher_model.eval()
    print('teacher model params')
    print(teacher_model.print_trainable_parameters())

    tokenizer = AutoTokenizer.from_pretrained(teacher_save_dir, trust_remote=True)
    processor = AutoProcessor.from_pretrained(teacher_save_dir)

    # dataset prepare
    def preprocess_func(example):
        MAX_LENGTH = 4096
        input_ids, attention_mask, labels = [], [], []
        url = example["message"][0]["conversation"][0]['url']
        caption = example["message"][0]["conversation"][1]['caption']
        
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful assistant in recognize math equations in either handwritten or printed text."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Recognize the equation in the image, write its LaTeX code between $$\n and \n$$"
                    },
                    {
                        "type": "image",
                        "image": url,
                        "resized_height": 128,
                        "resized_width": 128,
                    },
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": caption
                    }
                ]
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        img_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=img_inputs,
            padding=True,
            return_tensors='pt'
        )
        inputs = inputs.to('cuda')
        inputs = {key: value.tolist() for key, value in inputs.items()}
        instruction = inputs
        response = tokenizer(f'{caption}', add_special_tokens=False)
        input_ids = (
            instruction["input_ids"][0] + response['input_ids'] + [tokenizer.pad_token_id]
        )
        attention_mask = instruction['attention_mask'][0] + response['attention_mask'] + [1]
        labels = (
            [-100] * len(instruction['input_ids'][0])
            + response['input_ids']
            + [tokenizer.pad_token_id]
        )

        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        labels = torch.tensor(labels)

        inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])
        # 由 (1, h, w) 变换为 (h, w)
        inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)  
        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask, 
            "labels": labels,
            "pixel_values": inputs['pixel_values'], 
            "image_grid_thw": inputs['image_grid_thw']
        }
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    train_data = Dataset.from_json('ft_data_train.json')
    train_data = train_data.map(preprocess_func)
    print(train_data)

    # set up swanlab
    swanlab_callback = SwanLabCallback(
        project="Qwen2.5-VL-distill",
        experiment_name="qwen2.5-vl-crohme-2019",
        config={
            "model": "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
            "dataset": "https://disk.pku.edu.cn/anyshare/en-us/link/AAF10CCC4D539543F68847A9010C607139?_tb=none&expires_at=1970-01-01T08%3A00%3A00%2B08%3A00&item_type=&password_required=false&title=HMER%20Dataset&type=anonymous",
            "github": "https://github.com/Wooonster/HOCR",
            "prompt": "Recognize the equation in the image, write its LaTeX code between $$\n and \n$$",
            "train_data_number": len(train_data),
            "lora_rank": 64,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
        },
    )

    import os
    os.makedirs('./distill/student/', exist_ok=True)
    os.makedirs('./distill/saves/', exist_ok=True)

    # training args
    args = TrainingArguments(
        output_dir='./distill/student/',
        num_train_epochs=10,
        do_train=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        logging_steps=10,
        logging_first_step=5,
        report_to="none",
        save_strategy='epoch',
        save_total_limit=10,
        bf16=True,
        learning_rate=1e-4,
        lr_scheduler_type='cosine',
    )

    trainer = MyTrainer(
        model=model,
        teacher_model=teacher_model,
        args=args,
        train_dataset=train_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[swanlab_callback],
        optimizers=(None, None)
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./distill/saves/')
    trainer.save_state()