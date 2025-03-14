import numpy as np
import pandas as pd
import json
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.metrics.distance import edit_distance

DASH = '-' * 50
test_data_dir = './data/ft_data_test.json'
test_data = defaultdict()
with open(test_data_dir, 'r', encoding='utf-8') as f:
    data = json.laod(f)
    for d in data:
        test_data[d['message'][0]['conversation'][0]['url']] = d['message'][0]['conversation'][0]['caption']

def pre_process(txt):
    """
    Preprocess the text:
        1. check and remove "$$\t" and "\t$$" at the beginning and the end of pred_code and caption_code
        2. remove all " "
    """
    if txt.startswith("$$\t"):
        txt = txt[len("\t$$"):]
    if txt.endswith("\t$$"):
        txt = txt[:-len("\t$$")]
    txt = txt.replace(' ', '')
    return txt


def compute_bleu(preds, max_n=4):
    """
    Compute the BLEU score.
    1. match the url from `preds` and `test_data` to extract pred_code and caption_code
    2. preprocess the latex code text
    3. compute the BLEU scores

    args:
        preds (dict): prediction results from different models, 
            preds: {'url': 'predict latext code'}
        max_n (int):  maximum n-gram level, set default to 4

        global arg: test_data: {'url': 'caption latext code'}, the correct latext code

    return:
        float: average BLEU scores, ranging from 0 to 1
    """
    print(DASH + "Computing BLEU" + DASH)

    cc = SmoothingFunction()
    total_bleu = 0.0
    count = 0
    
    for url, pred_code in preds.items():
        if url not in test_data:
            continue
        ref_code = test_data[url]
        
        pred_processed = pre_process(pred_code)
        ref_processed = pre_process(ref_code)
        
        # 将预处理后的字符串转换为字符列表（字符级 tokenization）
        candidate_tokens = list(pred_processed)
        reference_tokens = list(ref_processed)
        
        # 使用平滑函数，防止因某些 n-gram 数量为 0 导致 log(0)错误
        weights = tuple([1.0 / max_n] * max_n)
        score = sentence_bleu([reference_tokens], candidate_tokens, weights=weights, smoothing_function=cc.method1)
        total_bleu += score
        count += 1
        
    average_bleu = total_bleu / count if count > 0 else 0.0
    print(f"Average BLEU score: {average_bleu:.4f}")
    # return average_bleu

def compute_exprate(preds):
    """
    Compute ExpRate score.
    1. match the url from `preds` and `test_data` to extract pred_code and caption_code
    2. preprocess the latex code text
    3. compute full exp rate
    4. compute <=1 exp rate
    5. compute <=2 exp rate
    
    args:
        preds (dict): prediction results from different models, 
            preds: {'url': 'predict latext code'}

        global arg: test_data: {'url': 'caption latext code'}, the correct latext code    

    return:
        tuple(float): full, <=1, <=2 exp rate
    """
    print(DASH + "Computing ExpRate" + DASH)

    # 根据 URL 匹配，构造预测和标签的对
    pairs = []
    for url, pred_code in preds.items():
        if url in test_data:
            caption_code = test_data[url]
            pred_processed = pre_process(pred_code)
            caption_processed = pre_process(caption_code)
            pairs.append((pred_processed, caption_processed))

    print(pairs)
    
    if not pairs:
        raise ValueError("No valid prediction-label pairs found. Please check preds and test_data.")
    
    # 用于统计匹配情况
    length = len(pairs)
    correct_count = 0
    correct_within_1 = 0
    correct_within_2 = 0
    
    # 遍历预测-标签对，统一将两者转换为字符序列（每个字符之间以空格分隔）
    for pred, gt in pairs:
        # 将字符串转换为单字符列表后再用空格连接
        pred_chars = ' '.join(list(pred))
        gt_chars = ' '.join(list(gt))

        # print((pred_chars, gt_chars))
        
        if pred_chars == gt_chars:
            correct_count += 1
        else:
            dist = edit_distance(pred_chars, gt_chars)
            if dist <= 1:
                correct_within_1 += 1
            if dist <= 2:
                correct_within_2 += 1

    full_exprate = (correct_count / length) * 100
    exprate_within_1 = ((correct_count + correct_within_1) / length) * 100
    exprate_within_2 = ((correct_count + correct_within_2) / length) * 100

    print(f"ExpRate: {full_exprate:.2f}%")
    print(f"ExpRate (<=1 error): {exprate_within_1:.2f}%")
    print(f"ExpRate (<=2 errors): {exprate_within_2:.2f}%")
    
    # return full_exprate, exprate_within_1, exprate_within_2


# test_data = {
#     "1": "$$\t\\frac { 3 2 5 } { 6 6 }\t$$",
#     "2": "$$\ta = 3 \\left( 4 - \\sqrt { 1 0 } \\right) \\sqrt { 5 } \\left( 1 4 0 \\sqrt { 1 0 } - 5 \\right)\t$$"
# }
# preds = {
#     "1": "$$\t\\frac { 325 } { 66 }\t$$",
#     "2": "$$\ta = 3 \\left( 4 - \\sqrt { 10 } \\right) \\sqrt { 5 } \\left( 140 \\sqrt { 10 } - 5 \\right)\t$$"
# }
# compute_bleu(preds)
# compute_exprate(preds)