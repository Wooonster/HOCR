import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用 tokenizers 并行性警告

import io
import cv2
import math
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import multiprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split  # 新增
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# 设置随机种子以保证可复现性
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed(5525)

" dataset preprocessings "
def prepare_datasets(data_pq_file, test_size=0.1, random_state=42):
    df = pd.read_parquet(data_pq_file)
    captions, img_names, img_bytes = df['formula'], df['filename'], df['image']
    if not len(captions) == len(img_names) == len(img_bytes):
        warnings.warn('Warning! Dataset may got errors.')

    # 将数据集划分为训练集和验证集
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)
     
    # 计算最大序列长度
    train_max_len = train_df['formula'].apply(lambda x: len(x)).max()
    val_max_len = val_df['formula'].apply(lambda x: len(x)).max()
    overall_max_len = max(train_max_len, val_max_len)
    print(f"Maximum sequence length in dataset: {overall_max_len}")
    
    # 设置 Positional Encoding 的 max_len
    global_max_len = max(1000, overall_max_len + 100)  # 加上缓冲区
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), global_max_len

" customized tokenizer "
# Preprocessing the dataset with custom tokenizer using Byte-Pair Encoding (BPE)
def custom_tokenizer(captions, dictionary_dir, save_tokenizer_dir):
    # 读取字典项
    dictionary = []
    with open(dictionary_dir, 'r') as f:
        data = f.readlines()
        dictionary.extend([line.strip() for line in data])  # 读取字典项

    # 初始化 BPE 分词器
    tokenizer = Tokenizer(BPE())

    # 训练分词器
    trainer = BpeTrainer(
        vocab_size=len(dictionary),
        special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace()])  # 使用空格进行分词
    tokenizer.train_from_iterator(captions, trainer=trainer)  # 在提供的 caption 上训练分词器

    # 保存分词器
    tokenizer.save(save_tokenizer_dir)
    print(f'New tokenizer trained and saved!')

    return tokenizer

'''
Add a preprocess to images

Convert the image to grayscale, black the background and white the formula
'''
class PreprocessImage:
    def __call__(self, image):
        # 转换为灰度图像
        img = image.convert('L')
        
        # 转换为 numpy 数组
        img_arr = np.array(img)
        
        # 应用二值化阈值，确保公式为白色，背景为黑色
        _, img_bin = cv2.threshold(img_arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 转换回 PIL 图像
        image = Image.fromarray(img_bin)
        return image

# Dataset class for loading the CROHME dataset with image paths and LaTeX expressions
class CROHMEDataset(Dataset):
    def __init__(self, df, tokenizer, transform=None, max_seq_len=1000):  # 增加 max_seq_len 参数
        self.images = df['image'].tolist()
        self.captions = df['formula'].tolist()
        self.img_names = df['filename'].tolist()
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_byte = self.images[idx]
        latex_expr = self.captions[idx]
        img_name = self.img_names[idx]

        if img_byte is None:
            raise FileNotFoundError(f"Image '{img_name}' doesn't exist.")

        img = io.BytesIO(img_byte)
        image = Image.open(img).convert("L")  # 转换为灰度图像
        if self.transform:
            image = self.transform(image)

        # 自定义 Tokenizer
        encoded = self.tokenizer.encode(latex_expr)
        # 手动添加特殊标记
        cls_id = self.tokenizer.token_to_id('[CLS]')
        sep_id = self.tokenizer.token_to_id('[SEP]')
        encoded_ids = [cls_id] + encoded.ids + [sep_id]
        
        # 截断序列
        if len(encoded_ids) > self.max_seq_len:
            encoded_ids = encoded_ids[:self.max_seq_len]
        
        latex_encoded = torch.tensor(encoded_ids, dtype=torch.long)

        return image, latex_encoded

" CollateFn class defined from top "
# 顶层定义的 CollateFn 类
class CollateFn:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        images, latex_exprs = zip(*batch)
        images = torch.stack(images, dim=0)  # 将所有图像张量堆叠成一个批次
        
        # 动态填充 LaTeX 令牌序列到批次中最长的序列
        max_length = max(len(expr) for expr in latex_exprs)
        padded_exprs = torch.full((len(latex_exprs), max_length), self.pad_id, dtype=torch.long)
        
        for i, expr in enumerate(latex_exprs):
            length = len(expr)
            padded_exprs[i, :length] = expr  # 将原始令牌复制到填充后的张量中
                
        return images, padded_exprs

# 修改后的 create_collate_fn 函数
def create_collate_fn(tokenizer):
    pad_id = tokenizer.token_to_id('[PAD]')
    return CollateFn(pad_id)


'''
Stacked DenseNet Encoder: defines a single block of the DenseNet architecture

use a bottleneck layer (conv1x1) to reduce the number of input channels before the conv3x3 layer.

Growth Rate: Determines how many channels are added after each block.
Dropout: Optional dropout for regularization.
'''
class DenseNetBone(nn.Module):
    def __init__(self, in_channels, growth_rate, bottleneck_width, dropout_rate=0.0):
        super().__init__()
        # Compute intermediate channels ensuring divisibility by 4
        inter_channels = int(growth_rate * bottleneck_width / 4) * 4

        # First batch normalization and convolution (bottleneck layer)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        # Second batch normalization and convolution
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        # Apply batch norm, ReLU, and first convolution
        out = self.conv1(F.relu(self.bn1(x)))
        # Apply dropout if specified
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        # Apply batch norm, ReLU, and second convolution
        out = self.conv2(F.relu(self.bn2(out)))
        # Apply dropout if specified
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        # Concatenate input and output along the channel dimension
        out = torch.cat([x, out], dim=1)
        return out

'''
DenseNet class using the DenseNetBone, with given number of stacking blocks.

num_blocks: Number of DenseNetBone in this DenseNet.
growth_rate: Number of channels to add per block.
input_channels: Number of input channels to the first block.
bottleneck_width and dropout_rate: Control the bottleneck layers and regularization.

Features: A list of DenseNetBlocks.
Transition Layer: Reduces the number of channels and spatial dimensions after the blocks.
Forward Pass: Sequentially applies each block and then the transition layer.
'''
class DenseNet(nn.Module):
    def __init__(self, num_blocks, growth_rate, input_channels, bottleneck_width=4, dropout_rate=0.0):
        super().__init__()
        self.num_blocks = num_blocks
        self.growth_rate = growth_rate
        self.dropout_rate = dropout_rate

        self.features = nn.ModuleList()
        num_channels = input_channels

        for i in range(num_blocks):
            block = DenseNetBone(
                in_channels=num_channels,
                growth_rate=growth_rate,
                bottleneck_width=bottleneck_width,
                dropout_rate=dropout_rate
            )
            self.features.append(block)
            num_channels += growth_rate

        # Optional Transition Layer to reduce dimensions
        self.transition = nn.Sequential(
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels // 2, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        num_channels = num_channels // 2  # Update channel count after transition

        self.num_channels = num_channels
    
    def forward(self, x):
        for block in self.features:
            x = block(x)
        x = self.transition(x)
        return x

'''
Stacked DenseNet Encoder with residual connections
'''
class StackedDenseNetEncoder(nn.Module):
    def __init__(self, num_densenets, num_blocks_per_dense, growth_rate, input_channels, hidden_dim, bottleneck_width=4, dropout_rate=0.4):
        super().__init__()
        self.num_densenets = num_densenets

        # Create a ModuleList to hold multiple DenseNets
        self.densenets = nn.ModuleList()
        self.residual_convs = nn.ModuleList()  # Convolutions to match dimensions for residual connections
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

            # Add a convolutional layer with stride=2 if the input and output channels differ
            if current_channels != densenet.num_channels:
                self.residual_convs.append(
                    nn.Conv2d(current_channels, densenet.num_channels, kernel_size=1, stride=2, bias=False)
                )
            else:
                # If channels match but spatial dimensions change, still downsample identity
                self.residual_convs.append(
                    nn.AvgPool2d(kernel_size=2, stride=2) if num_densenets > 1 else None
                )

            current_channels = densenet.num_channels  # Update channels for the next DenseNet

        # Final convolution to match the hidden_dim with the decoder
        self.conv_final = nn.Conv2d(current_channels, hidden_dim, kernel_size=1)

        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(hidden_dim)

    def forward(self, x):
        for idx, densenet in enumerate(self.densenets):
            identity = x  # Save input for residual connection

            x = densenet(x)  # Forward through DenseNet

            # Apply convolution to identity if necessary
            if self.residual_convs[idx] is not None:
                identity = self.residual_convs[idx](identity)

            # Add residual connection
            x = x + identity  # Element-wise addition

            x = F.relu(x)  # Apply activation after addition

        x = self.conv_final(x)
        x = self.pos_encoding(x)

        # Reshape to (batch_size, seq_len, hidden_dim)
        batch_size, channels, H, W = x.size()
        x = x.view(batch_size, channels, H * W)
        x = x.permute(0, 2, 1)
        return x

" 2D Positional Encoding "
class PositionalEncoding2D(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, tensor):
        B, C, H, W = tensor.size()
        device = tensor.device

        y_pos = torch.arange(H, device=device).unsqueeze(1).repeat(1, W)  # Shape: (H, W)
        x_pos = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1)

        y_pos = y_pos.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)  # Shape: (B, 1, H, W)
        x_pos = x_pos.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)

        div_term = torch.exp(torch.arange(0, C, 2, device=device) * -(math.log(10000.0) / C))
        pe_y = torch.zeros_like(tensor)
        pe_y[:, 0::2, :, :] = torch.sin(y_pos * div_term[:, None, None])   # Shape: (B, C//2, H, W)
        pe_y[:, 1::2, :, :] = torch.cos(y_pos * div_term[:, None, None])

        pe_x = torch.zeros_like(tensor)
        pe_x[:, 0::2, :, :] = torch.sin(x_pos * div_term[:, None, None])
        pe_x[:, 1::2, :, :] = torch.cos(x_pos * div_term[:, None, None])

        tensor = tensor + pe_y + pe_x
        return tensor

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):  # 将 max_len 设置为 1000
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # Shape: (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Shape: (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices
        
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

# Transformer decoder for generating LaTeX sequences from image features
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, max_len=1000):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=max_len)  # 设置 max_len=1000
        self.decoder_layer = nn.TransformerDecoderLayer(hidden_dim, num_heads)  # Basic transformer decoder layer
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)  # Output layer to predict next tokens

    def forward(self, encoder_outputs, tgt, tgt_mask):
        # Embed the target tokens (partial LaTeX sequences) and add positional encoding
        tgt_embedded = self.embedding(tgt)
        tgt_embedded = self.positional_encoding(tgt_embedded)
        
        # Pass the embedded tokens and encoder outputs through the decoder
        outputs = self.transformer_decoder(
            tgt_embedded.transpose(0, 1), 
            encoder_outputs.transpose(0, 1), 
            tgt_mask=tgt_mask
        )
        outputs = self.fc_out(outputs.transpose(0, 1))  # Final output layer to get predicted tokens
        
        return outputs


" Encoder using stacked DenseNet "
class ImageToLatexModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_layers=4, num_heads=8, max_len=1051):
        super().__init__()
        self.encoder = StackedDenseNetEncoder(
            num_densenets=3,
            num_blocks_per_dense=4,
            growth_rate=12,
            input_channels=1,  # Assuming grayscale images
            hidden_dim=hidden_dim,
            bottleneck_width=4,
            dropout_rate=0.1
        )
        self.decoder = TransformerDecoder(vocab_size, hidden_dim, num_layers, num_heads, max_len=max_len)
    
    def forward(self, x, tgt, tgt_mask):
        encoder_outputs = self.encoder(x)  # (batch_size, seq_len, hidden_dim)
        outputs = self.decoder(encoder_outputs, tgt, tgt_mask)
        return outputs

# Mixed precision training using PyTorch's automatic mixed precision (AMP)
scaler = GradScaler()

# Generate masks
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    return mask

def train_one_epoch(model, train_loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for images, latex_exprs in train_loader:
        images = images.to(device)
        latex_exprs = latex_exprs.to(device)
        
        optimizer.zero_grad()
        tgt_input = latex_exprs[:, :-1]
        tgt_output = latex_exprs[:, 1:]
        
        # Add target mask
        tgt_seq_len = tgt_input.size(1)
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)
        
        # Use autocast and scaler only if device is 'cuda'
        if device == 'cuda':
            with autocast():  # 移除 device_type 和 dtype 参数
                output = model(images, tgt_input, tgt_mask)
                loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # For CPU or other devices, do not use autocast or scaler
            output = model(images, tgt_input, tgt_mask)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(train_loader)

def evaluate(model, val_loader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for images, latex_exprs in val_loader:
            images = images.to(device)
            latex_exprs = latex_exprs.to(device)
            
            tgt_input = latex_exprs[:, :-1]
            tgt_output = latex_exprs[:, 1:]
            
            tgt_seq_len = tgt_input.size(1)
            tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)
            
            output = model(images, tgt_input, tgt_mask)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            
            epoch_loss += loss.item()
    return epoch_loss / len(val_loader)

# Beam search
def beam_search(model, image, tokenizer, beam_width=5, max_seq_len=1000):
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
                logits = output[:, -1, :]  # Get logits for the last token
                log_probs = torch.log_softmax(logits, dim=-1)
                
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
                
                for k in range(beam_width):
                    candidate_seq = seq + [topk_indices[0, k].item()]
                    candidate_score = score + topk_log_probs[0, k].item()
                    all_candidates.append((candidate_score, candidate_seq))
            
            # Select the best sequences
            ordered = sorted(all_candidates, key=lambda tup: tup[0], reverse=True)
            sequences = [seq for score, seq in ordered[:beam_width]]
            scores = [score for score, seq in ordered[:beam_width]]
            
            # Check for end token
            if all(seq[-1] == tokenizer.token_to_id('[SEP]') for seq in sequences):
                break
        
        best_sequence = sequences[0]
        return best_sequence[1:]  # Exclude the [CLS] token

# Test model on new images and save predictions to file
def make_predictions(model, tokenizer, test_folder, output_file, beam_width=5):
    model.eval()
    results = []
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5], std=[0.5]          # 单通道归一化
        )
    ])
    
    for img_name in sorted(os.listdir(test_folder)):
        img_path = os.path.join(test_folder, img_name)
        print(f"Processing {img_path}")
        try:
            image = Image.open(img_path).convert("L")  # 确保为灰度图像
            image = transform(image).unsqueeze(0).to(device)
            
            # Use beam search
            best_sequence = beam_search(model, image, tokenizer, beam_width=beam_width, max_seq_len=1000)
            decoded_latex = tokenizer.decode(best_sequence)
            results.append(f"{img_name}: {decoded_latex}")
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            results.append(f"{img_name}: ERROR")
    
    with open(output_file, 'w') as f:
        f.write("\n".join(results))
    print(f"Predictions saved to {output_file}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Handwritten Math Formula Recognition - Training with Validation')
    parser.add_argument('--data_pq_file', type=str, default='dataset/train_parquets/training_data.parquet', help='Path to the training data parquet file')
    parser.add_argument('--dictionary_dir', type=str, default='dataset/dictionary.txt', help='Path to the dictionary file')
    parser.add_argument('--saved_tokenizer_dir', type=str, default='dataset/whole_tokenizer.json', help='Path to save/load the custom tokenizer')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of transformer decoder layers')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')  # 增大 batch_size
    parser.add_argument('--num_epochs', type=int, default=40, help='Number of training epochs') 
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--test_size', type=float, default=0.1, help='Proportion of the dataset to include in the validation split')
    parser.add_argument('--random_state', type=int, default=5525, help='Random state for dataset splitting')

    args = parser.parse_args()

    data_pq_file = args.data_pq_file
    dictionary_dir = args.dictionary_dir
    saved_tokenizer_dir = args.saved_tokenizer_dir
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    num_heads = args.num_heads
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    test_size = args.test_size
    random_state = args.random_state

    # Ensure the directories exist
    os.makedirs(os.path.dirname(test_output_dir), exist_ok=True)
    os.makedirs('results/checkpoints', exist_ok=True)

    # Set up device for GPU/CPU usage
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Using device: {device}')

    # Extract data details and split into training and validation sets
    train_df, val_df, global_max_len = prepare_datasets(data_pq_file, test_size=test_size, random_state=random_state)
    captions = list(train_df['formula']) + list(val_df['formula'])
    
    # Check if the custom tokenizer exists, if not, create one
    if os.path.exists(saved_tokenizer_dir):
        print(f'Saved tokenizer found at {saved_tokenizer_dir}')
        tokenizer = Tokenizer.from_file(saved_tokenizer_dir)
    else:
        print(f'No saved tokenizer found, creating a new one.')
        tokenizer = custom_tokenizer(captions=captions, dictionary_dir=dictionary_dir, save_tokenizer_dir=saved_tokenizer_dir)

    # Transform pipeline for the images
    transform = transforms.Compose([
        PreprocessImage(),                 # Add preprocess
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5], std=[0.5]          # Change to single channel
        )
    ])

    # Create datasets
    train_dataset = CROHMEDataset(df=train_df, tokenizer=tokenizer, transform=transform, max_seq_len=global_max_len)
    val_dataset = CROHMEDataset(df=val_df, tokenizer=tokenizer, transform=transform, max_seq_len=global_max_len)

    # 根据系统的 CPU 核心数动态设置 num_workers
    num_workers = 16

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=create_collate_fn(tokenizer),
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if device == 'cuda' else False  # 持久化 workers 提高数据加载效率
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=create_collate_fn(tokenizer),
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if device == 'cuda' else False
    )

    # Initialize the model, optimizer, and loss function
    vocab_size = tokenizer.get_vocab_size()
    model = ImageToLatexModel(vocab_size, hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads, max_len=global_max_len).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'))

    # tensor board
    writer = SummaryWriter('runs/latex_recognition_experiment')

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # Training loop
    print('----------------------------------- Training starts -----------------------------------')
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion)
        val_loss = evaluate(model, val_dataloader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Save the model if it achieves a better validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_dir = "results/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            print(f"Model saved at epoch {epoch + 1}")
        
        # 3. 记录指标到 TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Plot the loss over epochs and save the plot
    print('----------------------------------- Plot the losses -----------------------------------')
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, 'o-', label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, 'o-', label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('results/training_validation_loss.png')

    writer.close()