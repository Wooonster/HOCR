import os
import io
import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tokenizers import Tokenizer
from torchvision import transforms
import numpy as np

# ---------------------------
# 模型定义部分
# ---------------------------

class DenseNetBone(nn.Module):
    def __init__(self, in_channels, growth_rate, bottleneck_width, dropout_rate=0.0):
        super().__init__()
        # 计算中间通道数，确保可以被4整除
        inter_channels = int(growth_rate * bottleneck_width / 4) * 4

        # 第一个批归一化层和1x1卷积（瓶颈层）
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        # 第二个批归一化层和3x3卷积
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.conv2(F.relu(self.bn2(out)))
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = torch.cat([x, out], dim=1)
        return out

class DenseNet(nn.Module):
    def __init__(self, num_blocks, growth_rate, input_channels, bottleneck_width=4, dropout_rate=0.0):
        super().__init__()
        self.num_blocks = num_blocks
        self.growth_rate = growth_rate
        self.dropout_rate = dropout_rate

        self.features = nn.ModuleList()
        num_channels = input_channels

        for _ in range(num_blocks):
            block = DenseNetBone(
                in_channels=num_channels,
                growth_rate=growth_rate,
                bottleneck_width=bottleneck_width,
                dropout_rate=dropout_rate
            )
            self.features.append(block)
            num_channels += growth_rate

        # 过渡层：减少通道数和空间维度
        self.transition = nn.Sequential(
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels // 2, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        num_channels = num_channels // 2
        self.num_channels = num_channels

    def forward(self, x):
        for block in self.features:
            x = block(x)
        x = self.transition(x)
        return x

class StackedDenseNetEncoder(nn.Module):
    def __init__(self, num_densenets, num_blocks_per_dense, growth_rate, input_channels, hidden_dim, bottleneck_width=4, dropout_rate=0.4):
        super().__init__()
        self.num_densenets = num_densenets

        self.densenets = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        current_channels = input_channels

        for _ in range(num_densenets):
            densenet = DenseNet(
                num_blocks=num_blocks_per_dense,
                growth_rate=growth_rate,
                input_channels=current_channels,
                bottleneck_width=bottleneck_width,
                dropout_rate=dropout_rate
            )
            self.densenets.append(densenet)

            if current_channels != densenet.num_channels:
                self.residual_convs.append(
                    nn.Conv2d(current_channels, densenet.num_channels, kernel_size=1, stride=2, bias=False)
                )
            else:
                self.residual_convs.append(
                    nn.AvgPool2d(kernel_size=2, stride=2) if num_densenets > 1 else None
                )

            current_channels = densenet.num_channels

        self.conv_final = nn.Conv2d(current_channels, hidden_dim, kernel_size=1)
        self.pos_encoding = PositionalEncoding2D(hidden_dim)

    def forward(self, x):
        for idx, densenet in enumerate(self.densenets):
            identity = x

            x = densenet(x)

            if self.residual_convs[idx] is not None:
                identity = self.residual_convs[idx](identity)

            x = x + identity
            x = F.relu(x)

        x = self.conv_final(x)
        x = self.pos_encoding(x)

        batch_size, channels, H, W = x.size()
        x = x.view(batch_size, channels, H * W)
        x = x.permute(0, 2, 1)
        return x

class PositionalEncoding2D(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, tensor):
        B, C, H, W = tensor.size()
        device = tensor.device

        y_pos = torch.arange(H, device=device).unsqueeze(1).repeat(1, W)
        x_pos = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1)

        y_pos = y_pos.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)
        x_pos = x_pos.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)

        div_term = torch.exp(torch.arange(0, C, 2, device=device) * -(math.log(10000.0) / C))
        pe_y = torch.zeros_like(tensor)
        pe_y[:, 0::2, :, :] = torch.sin(y_pos * div_term[:, None, None])
        pe_y[:, 1::2, :, :] = torch.cos(y_pos * div_term[:, None, None])

        pe_x = torch.zeros_like(tensor)
        pe_x[:, 0::2, :, :] = torch.sin(x_pos * div_term[:, None, None])
        pe_x[:, 1::2, :, :] = torch.cos(x_pos * div_term[:, None, None])

        tensor = tensor + pe_y + pe_x
        return tensor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(hidden_dim, num_heads)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, encoder_outputs, tgt, tgt_mask):
        tgt_embedded = self.embedding(tgt)
        tgt_embedded = self.positional_encoding(tgt_embedded)
        
        outputs = self.transformer_decoder(
            tgt_embedded.transpose(0, 1),
            encoder_outputs.transpose(0, 1),
            tgt_mask=tgt_mask
        )
        outputs = self.fc_out(outputs.transpose(0, 1))
        
        return outputs

class ImageToLatexModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_layers=4, num_heads=8):
        super().__init__()
        self.encoder = StackedDenseNetEncoder(
            num_densenets=3,
            num_blocks_per_dense=4,
            growth_rate=12,
            input_channels=1,  # 假设输入为灰度图像
            hidden_dim=hidden_dim,
            bottleneck_width=4,
            dropout_rate=0.1
        )
        self.decoder = TransformerDecoder(vocab_size, hidden_dim, num_layers, num_heads)
    
    def forward(self, x, tgt, tgt_mask):
        encoder_outputs = self.encoder(x)
        outputs = self.decoder(encoder_outputs, tgt, tgt_mask)
        return outputs

# ---------------------------
# 图像预处理部分
# ---------------------------

class PreprocessImage:
    def __call__(self, image):
        # 转换为灰度图
        img = image.convert('L')
        
        # 转换为 numpy 数组
        img_arr = np.array(img)
        
        # 应用二值化阈值，确保公式为白色，背景为黑色
        _, img_bin = cv2.threshold(img_arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 转换回 PIL 图像
        image = Image.fromarray(img_bin)
        return image

# ---------------------------
# Beam Search 函数
# ---------------------------

def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    return mask

def beam_search(model, image, tokenizer, beam_width=5, max_seq_len=100):
    model.eval()
    with torch.no_grad():
        encoder_outputs = model.encoder(image)
        device = image.device
        
        sequences = [[tokenizer.token_to_id('[CLS]')]]
        scores = [0.0]
        
        for _ in range(max_seq_len):
            all_candidates = []
            for i in range(len(sequences)):
                seq = sequences[i]
                score = scores[i]
                
                tgt_input = torch.tensor([seq], device=device)
                tgt_mask = generate_square_subsequent_mask(len(seq)).to(device)
                
                output = model.decoder(encoder_outputs, tgt_input, tgt_mask)
                logits = output[:, -1, :]  # 获取最后一个时间步的 logits
                log_probs = torch.log_softmax(logits, dim=-1)
                
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
                
                for k in range(beam_width):
                    candidate_seq = seq + [topk_indices[0, k].item()]
                    candidate_score = score + topk_log_probs[0, k].item()
                    all_candidates.append((candidate_score, candidate_seq))
            
            # 选择最好的序列
            ordered = sorted(all_candidates, key=lambda tup: tup[0], reverse=True)
            sequences = [seq for score, seq in ordered[:beam_width]]
            scores = [score for score, seq in ordered[:beam_width]]
            
            # 检查是否所有序列都以 [SEP] 结尾
            if all(seq[-1] == tokenizer.token_to_id('[SEP]') for seq in sequences):
                break
        
        best_sequence = sequences[0]
        return best_sequence[1:]  # 排除 [CLS] 标记

# ---------------------------
# 预测函数
# ---------------------------

def make_predictions(model, tokenizer, test_folder, output_file, device, beam_width=5):
    model.eval()
    results = []
    preprocess = PreprocessImage()
    transform = transforms.Compose([
        preprocess,
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5], std=[0.5]  # 单通道归一化
        )
    ])
    
    for img_name in sorted(os.listdir(test_folder)):
        img_path = os.path.join(test_folder, img_name)
        print(f"Processing {img_path}")
        try:
            image = Image.open(img_path).convert("L")  # 转换为灰度图像
            image = transform(image).unsqueeze(0).to(device)
            
            best_sequence = beam_search(model, image, tokenizer, beam_width=beam_width)
            decoded_latex = tokenizer.decode(best_sequence)
            results.append(f"{img_name}: {decoded_latex}")
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            results.append(f"{img_name}: ERROR")
    
    with open(output_file, 'w') as f:
        f.write("\n".join(results))
    print(f"Predictions saved to {output_file}")

# ---------------------------
# 主程序
# ---------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Handwritten Math Formula Recognition - Inference Script')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint (.pth file)')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to the custom tokenizer (.json file)')
    parser.add_argument('--test_folder', type=str, required=True, help='Path to the folder containing test images')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the prediction results')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer decoder layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    args = parser.parse_args()

    # 设置设备
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Using device: {device}')

    # 加载自定义分词器
    if os.path.exists(args.tokenizer):
        print(f'Loading tokenizer from {args.tokenizer}')
        tokenizer = Tokenizer.from_file(args.tokenizer)
    else:
        raise FileNotFoundError(f"Tokenizer file '{args.tokenizer}' not found.")

    vocab_size = tokenizer.get_vocab_size()

    # 初始化模型
    model = ImageToLatexModel(vocab_size, hidden_dim=args.hidden_dim, num_layers=args.num_layers, num_heads=args.num_heads).to(device)
    print('Model initialized.')

    # 加载模型权重
    if os.path.exists(args.checkpoint):
        print(f'Loading model weights from {args.checkpoint}')
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        print('Model weights loaded.')
    else:
        raise FileNotFoundError(f"Checkpoint file '{args.checkpoint}' not found.")

    # 执行预测
    make_predictions(model, tokenizer, args.test_folder, args.output_file, device, beam_width=5)
