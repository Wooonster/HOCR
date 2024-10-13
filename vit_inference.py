import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from tokenizers import Tokenizer
from transformers import ViTModel


# Define a function to generate masks
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    return mask

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x
    
# ViT Encoder
class ViTEncoder(nn.Module):
    def __init__(self, pretrained_model_name='google/vit-base-patch16-224-in21k', hidden_dim=768):
        super(ViTEncoder, self).__init__()
        self.encoder = ViTModel.from_pretrained(pretrained_model_name)
        self.hidden_dim = hidden_dim
        self.projection = nn.Linear(self.encoder.config.hidden_size, hidden_dim)

    def forward(self, x):
        outputs = self.encoder(pixel_values=x)
        last_hidden_state = outputs.last_hidden_state
        projected_state = self.projection(last_hidden_state)
        return projected_state

# Transformer Decoder
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
        outputs = self.transformer_decoder(tgt_embedded.transpose(0, 1), encoder_outputs.transpose(0, 1), tgt_mask=tgt_mask)
        outputs = self.fc_out(outputs.transpose(0, 1))
        return outputs

# Image to LaTeX Model
class ImageToLatexModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=768, num_layers=6, num_heads=8):
        super(ImageToLatexModel, self).__init__()
        self.encoder = ViTEncoder(hidden_dim=hidden_dim)
        self.decoder = TransformerDecoder(vocab_size, hidden_dim, num_layers, num_heads)

    def forward(self, x, tgt, tgt_mask):
        encoder_outputs = self.encoder(x)
        outputs = self.decoder(encoder_outputs, tgt, tgt_mask)
        return outputs

# Beam Search for Inference
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
                logits = output[:, -1, :]
                log_probs = torch.log_softmax(logits, dim=-1)
                
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
                
                for k in range(beam_width):
                    candidate_seq = seq + [topk_indices[0, k].item()]
                    candidate_score = score + topk_log_probs[0, k].item()
                    all_candidates.append((candidate_score, candidate_seq))
            
            ordered = sorted(all_candidates, key=lambda tup: tup[0], reverse=True)
            sequences = [seq for score, seq in ordered[:beam_width]]
            scores = [score for score, seq in ordered[:beam_width]]
            
            if all(seq[-1] == tokenizer.token_to_id('[SEP]') for seq in sequences):
                break
        
        best_sequence = sequences[0]
        return best_sequence[1:]  # Exclude the [CLS] token

# Make Predictions Function
def make_predictions(model, tokenizer, test_folder, output_file, beam_width=5):
    model.eval()
    results = []
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    for img_name in sorted(os.listdir(test_folder)):
        img_path = os.path.join(test_folder, img_name)
        print(f"Processing {img_path}")
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        
        best_sequence = beam_search(model, image, tokenizer, beam_width=beam_width)
        decoded_latex = tokenizer.decode(best_sequence)
        results.append(f"{img_name}: {decoded_latex}")
    
    with open(output_file, 'w') as f:
        f.write("\n".join(results))

# Main Inference Script
if __name__ == "__main__":
    # Path to the saved tokenizer and model
    saved_tokenizer_dir = 'crohme/train/custom_tokenizer.json'
    model_path = 'results/checkpoints/best_model.pth'
    test_img_base_dir = 'test'
    test_output_dir = 'results/test_results.txt'

    # Check if the custom tokenizer exists
    if os.path.exists(saved_tokenizer_dir):
        print(f'Saved tokenizer found at {saved_tokenizer_dir}')
        tokenizer = Tokenizer.from_file(saved_tokenizer_dir)
    else:
        raise FileNotFoundError(f"Tokenizer not found at {saved_tokenizer_dir}")

    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = tokenizer.get_vocab_size()
    model = ImageToLatexModel(vocab_size).to(device)
    
    # Load the saved model state dictionary
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Loaded model from {model_path}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    # Make predictions
    print("Making predictions on test data...")
    make_predictions(model, tokenizer=tokenizer, test_folder=test_img_base_dir, output_file=test_output_dir)
    print(f"Test results saved to {test_output_dir}")
