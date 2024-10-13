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

from torch.amp import autocast, GradScaler


" Extract images from pkl "
def extract_img(pkl_file: str, save_dir: str):
    """
    Extracts images from a pickle file and saves them as image files in the specified directory.
    
    Parameters:
        pkl_file (str): Path to the pickle file containing image data.
        save_dir (str): Directory where the extracted images will be saved.
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


" Merge image and labels to a csv "
def merge_hmer_or_crohme(images: str, caption: str, extract_img_dir: str, save_file: str):
    """
    Optimized function to read images from a pickle file and labels from a text file,
    then saves the matched image paths with their corresponding LaTeX labels into a CSV file.

    Parameters:
        images (str): Path to the pickle (.pkl) file containing image paths.
        caption (str): Path to the caption text file containing image-label pairs.
        save_file (str): Path to the output CSV file where the results will be saved.
    """
    
    # Load the image paths from the pickle file
    with open(images, 'rb') as f:
        image_list = pickle.load(f)
    """
    Load image paths into `image_list` from the pickle file.
    """

    # Create a dictionary for fast lookup: key is the image filename, value is the full path
    image_dict = {os.path.basename(img_path): img_path for img_path in image_list}
    """
    The `image_dict` allows for O(1) average-time complexity lookups for matching images.
    """

    # Read the caption file
    with open(caption, 'r', encoding='utf-8') as f:
        data = f.readlines()
    """
    Read all lines from the caption.txt file.
    """
    
    img_labels = []

    # Process each line in the caption file
    for line in data:
        """
        Loop through each line in the caption file.
        """
        
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
            # image_path = (extract_img_dir + '/' + image_dict[image_name])  # if extract_img_dir not None else image_dict[image_name]
            image_path = os.path.join(extract_img_dir, image_dict[image_name] + '.png')
            print(image_path)
            if os.path.isfile(image_path):
                img_labels.append([image_path, label])
            else:
                print(f"Warning: Image {image_name} not found in directory {extract_img_dir}. Skipping.")
        else:
            print(f"Warning: Image {image_name} not found in the pickle file. Skipping.")
    """
    The loop now efficiently finds image paths using dictionary lookups instead of iterating through a list.
    """

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
    
    # init tokenizer, using BPE
    tokenizer = Tokenizer(BPE())

    # train the tokenizer
    trainer = BpeTrainer(vocab=dictionary, special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace()])  # Use whitespace to segment words
    tokenizer.train_from_iterator(captions, trainer=trainer)  # Train the tokenizer on provided captions
    
    # save tokenizer
    tokenizer.save(save_tokenizer_dir)

    return tokenizer

# Dataset class for loading the CROHME dataset with image paths and LaTeX expressions
class CROHMEDataset(Dataset):
    def __init__(self, csv_file, tokenizer, transform=None, img_base_dir=None):
        self.data = pd.read_csv(csv_file)
        # self.data.dropna(subset=['latex'], inplace=True)  # delete Nan
        self.transform = transform
        self.tokenizer = tokenizer
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.img_base_dir = img_base_dir  # Base directory for image paths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Fetch image path and LaTeX expression from the CSV file
        img_path = self.data.iloc[idx, 0]
        # img_path = img_path if not self.img_base_dir else os.path.join(self.img_base_dir, img_path)
        latex_expr = self.data.iloc[idx, 1]

        # check image exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        # Load and transform the image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        '''
        Custom Tokenizer
        '''
        encoded = self.tokenizer.encode(latex_expr)
        latex_encoded = torch.tensor(encoded.ids, dtype=torch.long)

        '''
        BertTokenizer
        '''
        # # check latex formula is vaild
        # if not isinstance(latex_expr, str) or latex_expr.strip() == '':
        #     raise ValueError(f"Invalid LaTeX expression: {latex_expr}")
        
        # # change latex to a dicationary
        # try:
        #     latex_encoded = self.tokenizer.encode(latex_expr, add_special_tokens=True, max_length=500, truncation=True)
        #     latex_encoded = torch.tensor(latex_encoded, dtype=torch.long)
        # except Exception as e:
        #     raise ValueError(f"Error encoding LaTeX expression: {latex_expr}") from e
        
        return image, latex_encoded

# collate_fn for dynamic padding
def collate_fn(batch):
    images, latex_exprs = zip(*batch)
    images = torch.stack(images, dim=0)  # Stack all image tensors into a batch
    
    # Dynamically pad LaTeX token sequences to the longest sequence in the batch
    max_length = max(len(expr) for expr in latex_exprs)
    padded_exprs = torch.zeros(len(latex_exprs), max_length, dtype=torch.long)
    
    for i, expr in enumerate(latex_exprs):
        padded_exprs[i, :len(expr)] = expr  # Copy the original tokens to padded tensor
        
    return images, padded_exprs

# Vision Transformer (ViT) encoder model for extracting image features
class ViTEncoder(nn.Module):
    def __init__(self, pretrained_model_name='google/vit-base-patch16-224-in21k', hidden_dim=768):
        super(ViTEncoder, self).__init__()
        # Load the pretrained ViT model
        self.encoder = ViTModel.from_pretrained(pretrained_model_name)
        self.hidden_dim = hidden_dim
        
        # Linear projection layer to match transformer decoder's hidden dimension
        self.projection = nn.Linear(self.encoder.config.hidden_size, hidden_dim)

    def forward(self, x):
        # Pass the image tensor through the ViT encoder
        outputs = self.encoder(pixel_values=x)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        projected_state = self.projection(last_hidden_state)  # (batch_size, seq_len, hidden_dim)
        
        return projected_state
    
# Add a Sinusoidal Positional Encoding
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
        self.positional_encoding = PositionalEncoding(hidden_dim)  # apply sinusoidal positional encoding
        self.decoder_layer = nn.TransformerDecoderLayer(hidden_dim, num_heads)  # Basic transformer decoder layer
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)  # Output layer to predict next tokens

    def forward(self, encoder_outputs, tgt, tgt_mask):
        # Embed the target tokens (partial LaTeX sequences) and add positional encoding
        tgt_embedded = self.embedding(tgt)
        tgt_embedded = self.positional_encoding(tgt_embedded)
        
        # Pass the embedded tokens and encoder outputs through the decoder
        # tgt_embedded: (batch_size, tgt_seq_len, hidden_dim)
        # encoder_outputs: (batch_size, encoder_seq_len, hidden_dim)
        outputs = self.transformer_decoder(
            tgt_embedded.transpose(0, 1), 
            encoder_outputs.transpose(0, 1), 
            tgt_mask=tgt_mask
        )
        outputs = self.fc_out(outputs.transpose(0, 1))  # Final output layer to get predicted tokens
        
        return outputs

# Combined Image-to-LaTeX model with ViT encoder and transformer decoder
class ImageToLatexModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=768, num_layers=6, num_heads=8):
        super(ImageToLatexModel, self).__init__()
        self.encoder = ViTEncoder(hidden_dim=hidden_dim)  # ViT encoder
        self.decoder = TransformerDecoder(vocab_size, hidden_dim, num_layers, num_heads)  # Transformer decoder

    def forward(self, x, tgt, tgt_mask):
        encoder_outputs = self.encoder(x)  # Get image features
        outputs = self.decoder(encoder_outputs, tgt, tgt_mask)  # Generate LaTeX token sequences
        return outputs

# Mixed precision training using PyTorch's automatic mixed precision (AMP)
# Import the new amp module from torch
scaler = torch.amp.GradScaler()

# generate masks
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
        
        # add target mask
        tgt_seq_len = tgt_input.size(1)
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)
        
        # Use torch.amp.autocast instead of torch.cuda.amp.autocast
        with autocast('cuda'):
            # Forward pass
            output = model(images, tgt_input, tgt_mask)
            # Calculate loss
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        # Update weights using the optimizer
        scaler.step(optimizer)
        # Update the scale factor
        scaler.update()
        
        # Accumulate loss for this epoch
        epoch_loss += loss.item()
    
    # Return average loss over all batches
    return epoch_loss / len(train_loader)

# add a beam search to replace greedy search
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
        transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust mean and std if needed
    ])
    
    for img_name in sorted(os.listdir(test_folder)):
        img_path = os.path.join(test_folder, img_name)
        print(f"Processing {img_path}")
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        
        " change to use beam search "
        best_sequence = beam_search(model, image, tokenizer, beam_width=beam_width)
        decoded_latex = tokenizer.decode(best_sequence)
        results.append(f"{img_name}: {decoded_latex}")
        
        # with torch.no_grad():
        #     outputs_tokens = []
        #     # tgt_input = torch.tensor([[tokenizer.token_to_ids('[CLS]')]]).unsqueeze(
        #     tgt_input = torch.tensor([[tokenizer.token_to_id('[CLS]')]]).to(device)
            
        #     for _ in range(100):  # Limit sequence generation to 100 tokens
        #         tgt_seq_len = tgt_input.size(1)
        #         tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)

        #         output = model(image, tgt_input, tgt_mask)

        #         # get the prediction tokens
        #         next_token = output.argmax(dim=-1)[:, -1]
        #         outputs_tokens.append(next_token.item())
                
        #         if next_token.item() == tokenizer.token_to_id('[SEP]'):  # End sequence on [SEP]
        #             break

        #         tgt_input = torch.cat((tgt_input, next_token.unsqueeze(0)), dim=1)

        #     decoded_latex = tokenizer.decode(outputs_tokens)
        #     results.append(f"{img_name}: {decoded_latex}")
    
    with open(output_file, 'w') as f:
        f.write("\n".join(results))

# Main script for training and testing the model
if __name__ == "__main__":
    saved_tokenizer_dir = 'crohme/train/custom_tokenizer.json'
    caption_dir = 'crohme/train/caption.txt'
    dictionary_dir = 'crohme/dictionary.txt'
    training_img_pkl_dir = 'crohme/train/images.pkl'
    train_img_base_dir = 'crohme/train/extracted_img'
    mapping_csv = 'crohme/train/crohme_labels.csv'
    test_img_base_dir = 'test/'
    test_output_dir = 'results/test_results.txt'

    # train_img_base_dir is empty
    if not os.path.exists(train_img_base_dir) or os.path.exists(train_img_base_dir) and not os.listdir(train_img_base_dir):
        print('Extracting images...')
        extract_img(pkl_file=training_img_pkl_dir, save_dir=train_img_base_dir)
    else:
        print('No need to extract!')
    
    # mapping csv not exists
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
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Create dataset and data loader
    train_dataset = CROHMEDataset(mapping_csv, tokenizer=tokenizer, transform=transform, img_base_dir=train_img_base_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, collate_fn=collate_fn, pin_memory=True)

    # Set up device for GPU/CPU usage
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Using device: {device}')

    # Initialize the model, optimizer, and loss function
    vocab_size = tokenizer.get_vocab_size()
    model = ImageToLatexModel(vocab_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_loss = float('inf')
    losses = []

    # Training loop
    num_epochs = 20
    print('Training starts')
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = train_one_epoch(model, train_dataloader, optimizer, criterion)
        losses.append(epoch_loss)
        
        # Save the model if it achieves a better loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "results/checkpoints/best_model.pth")
            print(f"Model saved at epoch {epoch + 1}")
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
    # Plot the loss over epochs and save the plot
    plt.plot(range(1, num_epochs + 1), losses, 'o-')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('results/training_loss.png')
    
    # Testing the trained model on new test images
    print("Testing model on simple_test folder...")
    # state_dict = torch.load('results/checkpoints/best_model.pth', map_location=device)
    # saved_model = model.load
    make_predictions(model, tokenizer=tokenizer, test_folder=test_img_base_dir, output_file=test_output_dir)
    print("Test results saved to test_output.txt")

