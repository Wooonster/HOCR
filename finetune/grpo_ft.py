import os
os.environ["HF_DATASETS_CACHE"] = "./data/cache/"

import re
import logging
import torch
import json
import random
import transformers
import sympy as sp
from dataclasses import dataclass
from datetime import datetime
from sympy.parsing.latex import parse_latex
from transformers.trainer_utils import get_last_checkpoint
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info

from PIL import Image

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser
from trl.trainer.grpo_trainer import GRPOTrainer
from vlm_trainer.grpo_trainer import Qwen2VLGRPOTrainer
from peft import LoraConfig, TaskType, get_peft_model
import swanlab
from swanlab.integration.transformers import SwanLabCallback

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LENGTH = 4096

@dataclass
class ScriptArguments:
    dataset_id_or_path: str = './ft_data.json'
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Avoid duplicate handlers
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)
    

def setup_dataset(processor, tokenizer, dataset_id_or_path):
    # 1) 直接加载 JSON 数组文件，每条数据长这样：{"message":[{...}]}
    ds = load_dataset(
        "json",
        data_files=dataset_id_or_path,
    )["train"]  # Dataset 类型，每行 {'message': [...]}

    # 2) 划分 train/test
    split = ds.train_test_split(
        test_size=0.1, shuffle=True, seed=5525
    )
    train_ds = split["train"].shuffle(seed=525).select(range(2000))
    test_ds  = split["test"]

    # 3) preprocess：提取 prompt, image, answer
    def preprocess(example):
        # 拿到那条对话
        msg        = example["message"][0]
        image_path = msg["conversation"][0]["url"]
        answer     = msg["conversation"][1]["caption"]

        # 构造一次性的 messages，用来生成带模板的 prompt
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant in recognizing handwritten math equations."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Recognize the equation in the image; return only the code between <latex> and </latex>."
                    },
                    {
                        "type": "image",
                        "image": image_path,
                        "resized_height": 224,
                        "resized_width": 224,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": answer
                    }
                ],
            },
        ]

        # 用 processor 生成最终的字符串 prompt
        prompt_str = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # **关键：** 这里把路径读成 PIL.Image
    	# Fix the image path before loading

        old_base = "/root/autodl-tmp/HOCR/dataset"
        new_base = "/dataset/"
	
        if image_path.startswith(old_base):
            relative_path = os.path.relpath(image_path, old_base)
	    image_path = os.path.join(new_base, relative_path)

        img = Image.open(image_path).convert("RGB")

        return {
            "prompt": prompt_str,  # Trainer.compute_loss 会直接用这个字符串
            "image":  img,         # AutoProcessor 会正确识别 PIL.Image
            "answer": answer,      # 供 reward 函数使用
        }

    # 4) map 掉旧列，只留 prompt/image/answer 三列
    train_ds = train_ds.map(
        preprocess,
        remove_columns=train_ds.column_names,
    )
    test_ds = test_ds.map(
        preprocess,
        remove_columns=test_ds.column_names,
    )

    return train_ds, test_ds


def extract_latex(text):
    # 尝试匹配不同的标签对
    patterns = [
        (r"<latex>(.*?)</latex>", re.DOTALL),
        (r"<start_latex>(.*?)</end_latex>", re.DOTALL),
        (r"<start_latex>(.*?)<end_latex>", re.DOTALL),
        (r"\\\[(.*?)\\\]", re.DOTALL),
        (r"\\begin\{[^}]*\}(.*?)\\end\{[^}]*\}", re.DOTALL),
        (r"```latex\s*(.*?)\s*```", re.DOTALL),
    ]
    
    for pattern, flags in patterns:
        match = re.search(pattern, text, flags)
        if match:
            extracted = match.group(1).strip().replace(' ', '')
            return extracted

    return None


def extract_answer(text):
    text = text.split("<start_latex>")[1]
    text = text.split("<end_latex>")[0]
    return text.strip().replace(' ', '')


def strict_format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<latex>.*</latex>$"
    matches = [re.match(pattern, completion, flags=re.DOTALL) for completion in completions]
    return [0.5 if match else 0.0 for match in matches]


def stric_correct_reward_func(completions, answer, **kwargs):
    # logger.info(f"<---------completions------>{completions}")
    extracted_answers = [extract_latex(completion) for completion in completions]
    # logger.info(f"<---------extracted_completion------>{extracted_answers}")
    # logger.info(f"<---------answer------>{answer}")
    answer = [extract_answer(ans) for ans in answer]
    # logger.info(f"<---------extract_answer------>{answer}")
    return [2.0 if ea == a else 0.0 for ea, a in zip(extracted_answers, answer)]


def soft_correct_reward_func(completions, answer, **kwargs):
    """
    计算软匹配奖励：如果解析后两者数学表达式相等，则返回1.0分，否则返回0.0分。
    
    使用 Sympy 的 LaTeX 解析器 parse_latex 将生成的 LaTeX 与标准答案解析为数学表达式。
    如果解析结果为 Equality（方程），则分别比较左右两侧。
    否则直接使用符号计算比较二者是否数学上相等。
    """
    scores = []

    extracted_answers = [extract_latex(completion) for completion in completions]
    answer = [extract_answer(ans) for ans in answer]
    for resp_latex, ans_latex in zip(extracted_answers, answer):
        # ans_latex = extract_latex(a)
        if resp_latex is None or ans_latex is None:
            scores.append(0.0)
        else:
            try:
                expr_resp = parse_latex(resp_latex)
                expr_ans = parse_latex(ans_latex)
                # 如果两者均为 Equality 对象，则分别比较左右两侧
                if isinstance(expr_resp, sp.Equality) and isinstance(expr_ans, sp.Equality):
                    diff_lhs = sp.simplify(expr_resp.lhs - expr_ans.lhs)
                    diff_rhs = sp.simplify(expr_resp.rhs - expr_ans.rhs)
                    if diff_lhs == 0 and diff_rhs == 0:
                        scores.append(1.0)
                    else:
                        scores.append(0.0)
                else:
                    # 否则，直接比较整个表达式
                    if sp.simplify(expr_resp - expr_ans) == 0:
                        scores.append(1.0)
                    else:
                        scores.append(0.0)
            except Exception as e:
                # 如果解析出错，则认为该输出不正确
                # print("Error when parsing latex:", e)
                scores.append(0.0)
    return scores

def length_penalty_reward(completions, max_words: int = 50, **kwargs):
    """
    只做长度惩罚（超长时线性递减），不再重复做 <latex>…</latex> 检查。
    
    - 正常：<= max_words 单词 → reward=1.0
    - 超长：score = max(0, 1 - overshoot/total_words)
    - 不含 latex 块：score=0.0 （由 strict_format_reward_func 单独保证格式）
    """
    latex_pat = re.compile(r'<latex>(.*?)</latex>', re.DOTALL)
    rewards = []
    
    for comp in completions:
        m = latex_pat.search(comp)
        if not m:
            # 格式检查交给 strict_format_reward_func，不在这里重复
            rewards.append(0.0)
            continue
        
        # 去掉 latex 块后统计“正文”单词数
        outside = latex_pat.sub('', comp).strip()
        words = [w for w in outside.split() if w]
        
        if len(words) <= max_words:
            rewards.append(1.0)
        else:
            overshoot = len(words) - max_words
            score = max(0.0, 1.0 - overshoot / len(words))
            rewards.append(score)
    
    return rewards


def grpo(model_args, script_args, training_args):
    # logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16, 
        # attn_implementation="flash_attention_2",
        # device_map='auto',
    )
    model.to(DEVICE)
    
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    processor = AutoProcessor.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    model.enable_input_require_grads()   # 开启梯度检查点时(training_args.gradient_checkpointing=True,)要执行该方法

    training_dataset, test_dataset = setup_dataset(processor, tokenizer, script_args.dataset_id_or_path)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "gate_proj", ],
        task_type="CAUSAL_LM",
    )

    # load swanlab
    swanlab_callback = SwanLabCallback(
        project="HOCR-GRPO",
        experiment_name="qwen2.5vl-3b-instruct-grpo",
        config={
            "model": "https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct",
            "dataset": "",
            "github": "https://github.com/Wooonster/HOCR",
            "prompt": "",
            "train_data_number": len(training_dataset),
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
        },
    )

    # init grpo trainer
    # Qwen2VLGRPOTrainer
    trainer = Qwen2VLGRPOTrainer(
        model=model,
        reward_funcs=[
            strict_format_reward_func, stric_correct_reward_func, soft_correct_reward_func, length_penalty_reward,
        ],
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=test_dataset,
        peft_config=lora_config,
        callbacks=[swanlab_callback,],
    )

    # # Check for last checkpoint
    # last_checkpoint = get_last_checkpoint(training_args.output_dir)
    # if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    #     logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # # Train the model
    # logger.info(
    #     f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    # )
    # train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    
    train_result = trainer.train()
    
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(training_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Train the model
    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )

    logger.info("*** Training complete! ***")

def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # training_args.model_init_kwargs = {"torch_dtype": torch.bfloat16}

    logger.info(f"Model   : {model_args}")
    logger.info(f"Dataset : {script_args}")
    logger.info(f"Training: {training_args}")

    # Run the main training loop
    grpo(model_args, script_args, training_args)

if __name__ == "__main__":
    main()
