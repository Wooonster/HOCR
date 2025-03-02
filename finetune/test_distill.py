import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoTokenizer, DataCollatorForSeq2Seq,
    Qwen2_5_VLForConditionalGeneration, AutoProcessor, 
    Trainer, TrainingArguments
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from qwen_vl_utils import process_vision_info

tokenizer = AutoTokenizer.from_pretrained('./distill/student/checkpoint-160/', trust_remote=True)
processor = AutoProcessor.from_pretrained('./vl3b/')

def predict(messages, model):
    # 准备推理
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 生成输出
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

def compute_exprate(res):
    from Levenshtein import distance
    length = len(res)
    correct_count = 0
    correct_within_1 = 0
    correct_within_2 = 0

    for pred, gt in res:
        # pred = list(pred[2:-2]) # 去掉 $$
        # gt = list(gt[3:-3])

        pred = pred[2:-2] # 去掉 $$
        gt = gt[3:-3] # 去掉 $$ 和 \n
        if pred == gt:
            correct_count += 1
        else:
            dist = distance(pred, gt)
            if dist <= 1:
                correct_within_1 += 1
            if dist <= 2:
                correct_within_2 += 1
    # 计算各项指标
    exprate = (correct_count / length) * 100
    exprate_within_1 = ((correct_count + correct_within_1) / length) * 100
    exprate_within_2 = ((correct_count + correct_within_2) / length) * 100

    print(f"ExpRate: {exprate:.2f}%")
    print(f"ExpRate (≤1 error): {exprate_within_1:.2f}%")
    print(f"ExpRate (≤2 errors): {exprate_within_2:.2f}%")


if __name__ == '__main__':
    # load student model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        './distill/student/checkpoint-160/',  torch_dtype=torch.float16, 
        attn_implementation="flash_attention_2", device_map="auto"
    )
    
    # 读取测试数据
    with open("ft_data_test.json", "r") as f:
        test_dataset = json.load(f)

    test_image_list = []
    results = []
    for item in test_dataset:
        url = item["message"][0]["conversation"][0]['url']
        caption = item["message"][0]["conversation"][1]['caption']
        
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
                        "resized_height": 280,
                        "resized_width": 280,
                    },
                ]
            }
        ]
        
        response = predict(messages, model)
        messages.append({"role": "assistant", "content": f"{response}"})
        results.append([messages[-1]['content'], caption])
        print(f'messages: {messages[-1]['content'][2:-2]}; ground-truth: {caption[3:-3]}')

    # 计算得分
    compute_exprate(results)