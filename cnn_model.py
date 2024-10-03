import os
import pickle
import cv2
import math
import heapq
import xml.etree.ElementTree as ET
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from tokenizers import Tokenizer, decoders, pre_tokenizers, processors, trainers
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from transformers import BertConfig, BertTokenizer, ViTModel
from torch.utils.data import DataLoader, Dataset
from torchvision.models import densenet121
from torch.amp import autocast, GradScaler

# Extract images from pkl
def extract_img(pkl_file: str, save_dir: str):
    """
    Extracts images from a pickle file and saves them as image files in the specified directory.
    """
    # Load the pickle file
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    # Ensure the output directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Iterate through the dictionary items
    for image_name, image_array in data.items():
        # Ensure image_name has a proper extension
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_name += '.png'  # Default to .png if no extension
        
        # Construct the full path for saving the image
        output_path = os.path.join(save_dir, image_name)
        
        # Convert image_array to an appropriate data type if needed
        if image_array.dtype != 'uint8':
            image_array = image_array.astype('uint8')
        
        # Save the image using OpenCV
        success = cv2.imwrite(output_path, image_array)
        if success:
            print(f"Saved image: {output_path}")
        else:
            print(f"Failed to save image: {output_path}")

# Merge image and labels to a csv
def merge_hmer_or_crohme(images: str, caption: str, extract_img_dir: str, save_file: str):
    """
    Reads images from a pickle file and labels from a text file,
    then saves the matched image paths with their corresponding LaTeX labels into a CSV file.
    """
    # Load the image paths from the pickle file
    with open(images, 'rb') as f:
        image_list = pickle.load(f)
    
    # Create a dictionary for fast lookup: key is the image filename, value is the full path
    image_dict = {os.path.basename(img_path): img_path for img_path in image_list}
    
    # Read the caption file
    with open(caption, 'r', encoding='utf-8') as f:
        data = f.readlines()
    
    img_labels = []

    # Process each line in the caption file
    for line in data:
        # Strip any leading/trailing whitespace/newlines
        line = line.strip()
        
        # Check for tab separation and split accordingly
        if '\t' in line:
            image_name, label = line.split('\t', 1)  # Split by the first tab
        else:
            # Skip if improperly formatted
            continue
        
        # Use the dictionary to find the matching image path
        if image_name in image_dict:
            image_path = os.path.join(extract_img_dir, image_dict[image_name] + '.png')
            if os.path.isfile(image_path):
                img_labels.append([image_path, label])
            else:
                print(f"Warning: Image {image_name} not found in directory {extract_img_dir}. Skipping.")
        else:
            print(f"Warning: Image {image_name} not found in the pickle file. Skipping.")
    
    # Create a DataFrame from the list of image-label pairs
    df = pd.DataFrame(img_labels, columns=['Image Path', 'Label'])

    # Save the DataFrame to a CSV file without index
    df.to_csv(save_file, index=False)
    print(f"CSV file saved as {save_file}")

# Preprocessing the dataset with custom tokenizer using Byte-Pair Encoding (BPE)
def custom_tokenizer(caption_dir, dictionary_dir, save_tokenizer_dir):
    captions = []
    with open(caption_dir, 'r') as f:
        data = f.readlines()
        for line in data:
            line = line.strip()
            if '\t' in line:
                _, label = line.split('\t', 1)  # Split by tab to get caption label
            else:
                continue  # Skip lines with improper formatting
            captions.append(label)

    dictionary = []
    with open(dictionary_dir, 'r') as f:
        data = f.readlines()
        dictionary.extend([line.strip() for line in data])  # Read in dictionary items
    
    # Initialize tokenizer using BPE
    tokenizer = Tokenizer(BPE())
    
    # Train the tokenizer
    trainer = BpeTrainer(vocab=dictionary, special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace()])  # Use whitespace to segment words
    tokenizer.train_from_iterator(captions, trainer=trainer)  # Train the tokenizer on provided captions
    
    # Save tokenizer
    tokenizer.save(save_tokenizer_dir)

    return tokenizer

'''
Add a preprocess to images

Convert the image to grayscale, black the background and white the formula
'''
class PreprocessImage:
    def __call__(self, image):
        # graysacle
        img = image.convert(':')

        # to numpy
        img_arr = np.array(img)

        # 应用二值化阈值，确保公式为白色，背景为黑色
        # THRESH_BINARY_INV 将公式（通常为黑色）转换为白色，背景为黑色
        _, img_bin = cv2.threshold(img_arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        image = Image.fromarray(img_bin)
        return image


# Dataset class for loading the CROHME dataset with image paths and LaTeX expressions
class CROHMEDataset(Dataset):
    def __init__(self, csv_file, tokenizer, transform=None, img_base_dir=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.tokenizer = tokenizer
        self.img_base_dir = img_base_dir  # Base directory for image paths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Fetch image path and LaTeX expression from the CSV file
        img_path = self.data.iloc[idx, 0]
        latex_expr = self.data.iloc[idx, 1]

        # Check if image exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        # Load and transform the image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Custom Tokenizer
        encoded = self.tokenizer.encode(latex_expr)
        # Manually add special tokens
        cls_id = self.tokenizer.token_to_id('[CLS]')
        sep_id = self.tokenizer.token_to_id('[SEP]')
        encoded_ids = [cls_id] + encoded.ids + [sep_id]
        latex_encoded = torch.tensor(encoded_ids, dtype=torch.long)
        
        return image, latex_encoded
    
# # 顶层定义的 collate_fn
# def collate_fn(batch, pad_id):
#     images, latex_exprs = zip(*batch)
#     images = torch.stack(images, dim=0)  # Stack all image tensors into a batch
    
#     # Dynamically pad LaTeX token sequences to the longest sequence in the batch
#     max_length = max(len(expr) for expr in latex_exprs)
#     padded_exprs = torch.full((len(latex_exprs), max_length), pad_id, dtype=torch.long)
    
#     for i, expr in enumerate(latex_exprs):
#         length = len(expr)
#         padded_exprs[i, :length] = expr  # Copy the original tokens to padded tensor
            
#     return images, padded_exprs


# # collate_fn for dynamic padding
# def create_collate_fn(tokenizer):
#     pad_id = tokenizer.token_to_id('[PAD]')
#     return lambda batch: collate_fn(batch, pad_id)
#     # pad_id = tokenizer.token_to_id('[PAD]')
#     # return collate_fn

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


" CNN Encoder using DenseNet121 "
class CNNEncoder(nn.Module):
    def __init__(self, hidden_dim=256):
        super(CNNEncoder, self).__init__()
        # Load pretrained DenseNet
        densenet = densenet121(pretrained=True)
        # Remove the classification layer
        self.cnn = nn.Sequential(*list(densenet.features.children()))
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(1024, hidden_dim)
        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_dim)

    def forward(self, x):
        # x shape: (batch_size, 3, H, W)
        features = self.cnn(x)  # Shape: (batch_size, hidden_dim, H', W')
        B, C, H, W = features.size()
        features = features.view(B, C, -1)  # Shape: (batch_size, channels, H'*W')
        features = features.permute(0, 2, 1)  # Shape: (batch_size, seq_len, channels)
        features = self.fc(features)  # Project to hidden_dim
        features = self.positional_encoding(features)  # Shape: (batch_size, seq_len, hidden_dim)
        return features
    
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

            # Add a convolutional layer if the input and output channels differ
            if current_channels != densenet.num_channels:
                self.residual_convs.append(
                    nn.Conv2d(current_channels, densenet.num_channels, kernel_size=1, bias=False)
                )
            else:
                self.residual_convs.append(None)  # No need for convolution if channels match

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

# # Combined Image-to-LaTeX model with CNN encoder and transformer decoder
# class ImageToLatexModel(nn.Module):
#     def __init__(self, vocab_size, hidden_dim=256, num_layers=4, num_heads=8):
#         super(ImageToLatexModel, self).__init__()
#         self.encoder = CNNEncoder(hidden_dim=hidden_dim)
#         self.decoder = TransformerDecoder(vocab_size, hidden_dim, num_layers, num_heads)

#     def forward(self, x, tgt, tgt_mask):
#         encoder_outputs = self.encoder(x)  # Get image features
#         outputs = self.decoder(encoder_outputs, tgt, tgt_mask)  # Generate LaTeX token sequences
#         return outputs


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
            with autocast():
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
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225]    # ImageNet stds
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

# Main script for training and testing the model
if __name__ == "__main__":
    saved_tokenizer_dir = 'dataset/crohme/train/custom_tokenizer.json'
    caption_dir = 'dataset/crohme/train/caption.txt'
    dictionary_dir = 'dataset/crohme/crohme_dictionary.txt'
    training_img_pkl_dir = 'dataset/crohme/train/images.pkl'
    train_img_base_dir = 'dataset/crohme/train/extracted_img'
    mapping_csv = 'dataset/crohme/crohme_labels.csv'
    test_img_base_dir = 'test/imgs'
    test_output_dir = 'results/test_results.txt'

    # Ensure the directories exist
    os.makedirs(train_img_base_dir, exist_ok=True)
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

    # Extract images if necessary
    if not os.path.exists(train_img_base_dir) or not os.listdir(train_img_base_dir):
        print('Extracting images...')
        extract_img(pkl_file=training_img_pkl_dir, save_dir=train_img_base_dir)
    else:
        print('No need to extract!')
    
    # Create mapping csv if necessary
    if not os.path.exists(mapping_csv):
        print('Creating mapping csv...')
        merge_hmer_or_crohme(images=training_img_pkl_dir, caption=caption_dir, extract_img_dir=train_img_base_dir, save_file=mapping_csv)
    else:
        print('No need to create mapping csv!')
    
    # Check if the custom tokenizer exists, if not, create one
    if os.path.exists(saved_tokenizer_dir):
        print(f'Saved tokenizer found at {saved_tokenizer_dir}')
        tokenizer = Tokenizer.from_file(saved_tokenizer_dir)
    else:
        print(f'No saved tokenizer found, creating a new one.')
        tokenizer = custom_tokenizer(caption_dir=caption_dir, dictionary_dir=dictionary_dir, save_tokenizer_dir=saved_tokenizer_dir)

    # Transform pipeline for the images
    transform = transforms.Compose([
        PreprocessImage(),                 # Add preprocess
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],  # ImageNet means
        #     std=[0.229, 0.224, 0.225]    # ImageNet stds
        # )  
        transforms.Normalize(
            mean=[0.5], std=[0.5]          # Change to single channel
        )
    ])

    # Create dataset and data loader
    train_dataset = CROHMEDataset(mapping_csv, tokenizer=tokenizer, transform=transform, img_base_dir=train_img_base_dir)
    # collate_fn = create_collate_fn(tokenizer)
    # train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, collate_fn=collate_fn, pin_memory=True)
    # 创建 DataLoader 时使用顶层 CollateFn 类
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8,
        collate_fn=create_collate_fn(tokenizer),
        pin_memory=True
    )

    # Initialize the model, optimizer, and loss function
    vocab_size = tokenizer.get_vocab_size()
    model = ImageToLatexModel(vocab_size, hidden_dim=256, num_layers=8, num_heads=8).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'))

    best_loss = float('inf')
    losses = []

    # Training loop
    num_epochs = 100
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
