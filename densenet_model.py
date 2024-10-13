import os
import io
import cv2
import math
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
from torch.utils.data import DataLoader, Dataset
from torchvision.models import densenet121
from torch.amp import autocast, GradScaler


" dataset preprocessings "
def prepare_datasets(data_pq_file):
    df = pd.read_parquet(data_pq_file)
    captions, img_names, img_bytes = df['formula'], df['filename'], df['image']
    assert len(captions) == len(img_names) == len(img_bytes), 'dataset parquet got errors'
    return df

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
        # graysacle
        img = image.convert('L')  # 'L' 转换为灰度图像
        
        # 转换为 numpy 数组
        img_arr = np.array(img)
        
        # 应用二值化阈值，确保公式为白色，背景为黑色
        _, img_bin = cv2.threshold(img_arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 转换回 PIL 图像
        image = Image.fromarray(img_bin)
        return image

# Dataset class for loading the CROHME dataset with image paths and LaTeX expressions
class CROHMEDataset(Dataset):
    def __init__(self, df, tokenizer, transform=None):
        self.images = df['image'].tolist()
        self.captions = df['formula'].tolist()
        self.img_names = df['filename'].tolist()
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_byte = self.images[idx]
        latex_expr = self.captions[idx]
        img_name = self.img_names[idx]

        if img_byte is None:
            raise FileNotFoundError(f"Image '{img_name}' doesn't exist.")

        img = io.BytesIO(img_byte)
        image = Image.open(img).convert("L")  # 根据前面的修正，转换为灰度图像
        if self.transform:
            image = self.transform(image)

        # 自定义 Tokenizer
        encoded = self.tokenizer.encode(latex_expr)
        # 手动添加特殊标记
        cls_id = self.tokenizer.token_to_id('[CLS]')
        sep_id = self.tokenizer.token_to_id('[SEP]')
        encoded_ids = [cls_id] + encoded.ids + [sep_id]
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
DenseNet class using the DenseNetBlock, with given number of stacking blocks.

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
    
" 2D Positional Encoding"
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
    def __init__(self, d_model, max_len=500):
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
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)  # Apply sinusoidal positional encoding
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
    def __init__(self, vocab_size, hidden_dim=256, num_layers=4, num_heads=8):
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
        self.decoder = TransformerDecoder(vocab_size, hidden_dim, num_layers, num_heads)
    
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
            with autocast(device_type='cuda', dtype=torch.float16):
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

# Beam search
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
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        
        # Use beam search
        best_sequence = beam_search(model, image, tokenizer, beam_width=beam_width)
        decoded_latex = tokenizer.decode(best_sequence)
        results.append(f"{img_name}: {decoded_latex}")
    
    with open(output_file, 'w') as f:
        f.write("\n".join(results))


if __name__ == '__main__':
    data_pq_file = 'dataset/hmer_train.parquet'
    dictionary_dir = 'dataset/dictionary.txt'
    saved_tokenizer_dir = 'dataset/custom_tokenizer.json'
    test_img_base_dir = 'test/imgs/'
    test_output_dir = 'results/test_results_densenet.txt'

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

    # extract data details
    data_df = prepare_datasets(data_pq_file)
    captions = list(data_df['formula'])
    
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

    # Create dataset and data loader
    train_dataset = CROHMEDataset(df=data_df, tokenizer=tokenizer, transform=transform)

    # 根据系统的 CPU 核心数动态设置 num_workers
    num_workers = min(8, multiprocessing.cpu_count())
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=create_collate_fn(tokenizer),
        pin_memory=True if device == 'cuda' else False
    )

    # Initialize the model, optimizer, and loss function
    vocab_size = tokenizer.get_vocab_size()
    model = ImageToLatexModel(vocab_size, hidden_dim=256, num_layers=6, num_heads=8).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'))

    best_loss = float('inf')
    losses = []

    # Training loop
    num_epochs = 30
    print('Training starts')
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = train_one_epoch(model, train_dataloader, optimizer, criterion)
        losses.append(epoch_loss)
        
        # Save the model if it achieves a better loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint_dir = "results/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            print(f"Model saved at epoch {epoch + 1}")
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
    # Plot the loss over epochs and save the plot
    plt.plot(range(1, num_epochs + 1), losses, 'o-')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('results/training_loss.png')

    # Testing the trained model on new test images
    print("Testing model on test folder...")
    # Load the best model
    checkpoint_path = os.path.join("results", "checkpoints", "best_model.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    make_predictions(model, tokenizer=tokenizer, test_folder=test_img_base_dir, output_file=test_output_dir)
    print(f"Test results saved to {test_output_dir}")