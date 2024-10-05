# HOCR: Handwriting OCR project

## Architecture

### DenseNet + Transformer decoder

The model uses a stacked DenseNet, with residual connections and positional embedding to store spatial informations, and a Transformer decoder to predict the $\LaTeX$ formula.

The model comprises two main components:

- **Encoder**: A stacked DenseNet architecture that processes input images to extract meaningful features.
- **Decoder**: A Transformer-based decoder that translates the extracted features into LaTeX code.

#### Detailed Architecture
1. DenseNet Encoder

- DenseNetBone: The fundamental building block, consisting of two convolutional layers with bottleneck layers and optional dropout for regularization.
- DenseNet: Stacks multiple DenseNetBone blocks and includes transition layers to reduce spatial dimensions and channel numbers.
- StackedDenseNetEncoder: Combines multiple DenseNet modules with residual connections, culminating in a final convolution and 2D positional encoding to prepare features for the decoder.

2. Transformer Decoder

- Embedding Layer: Converts token IDs into dense vectors.
- Positional Encoding: Adds sinusoidal positional information to embeddings.
- TransformerDecoderLayer: Comprises multi-head attention and feed-forward neural networks.
- Final Linear Layer: Maps decoder outputs to vocabulary size for token prediction.

### Workflow

- Image Preprocessing: Converts input images to grayscale, applies binary thresholding to ensure white formulas on black backgrounds, resizes to 224x224, and normalizes.
- Feature Extraction: The DenseNet encoder processes the preprocessed image to extract feature maps.
- Sequence Generation: The Transformer decoder generates LaTeX tokens based on the encoder's output, utilizing beam search to optimize predictions.


##### Previous Failed Architecture: ~~ViT~~ + Transformer

*Reason why ViT doesn't work well*: 

Domain differences in pre-training ViT models: ViT (Vision Transformer) is usually pre-trained on large-scale natural image datasets like ImageNet. Your dataset, on the other hand, is an image of handwritten mathematical formulas, which is very different from natural images in terms of feature distribution. A pre-trained ViT model may not be able to extract valid features, resulting in encoder output that is meaningless to the decoder.

Lack of low-level feature extraction: handwritten formula images have complex local features, such as stroke and symbol details. viT slices the image directly into patches and then performs a global self-attention mechanism, which may not be able to capture these critical local features.


## Datasets and preprocessing

The dataset combines many different datasets, with a total amount of over 240K images. 

The images are onverted to grayscale, applied binary thresholding to ensure white formulas on black backgrounds, resized to 224x224, and normalized.

### Dataset Structure
The project expects the dataset to be in a Parquet file (.parquet) with the following columns:

- formula: The LaTeX expression of the handwritten formula.
- filename: The name of the image file.
- image: The binary content of the image.

## Installation

### Prerequisites

Python: 3.7 or higher
PyTorch: 1.8.0 or higher
CUDA: (Optional) For GPU acceleration


### init

The versions of the packages listed in `requirements.txt` are not guaranteed to work.

`pip install -r requirements.txt`

If requirements.txt is not provided, install the necessary packages manually:

`pip install torch torchvision tokenizers pillow opencv-python matplotlib numpy pandas tqdm`

## Run

### Dataset preparing

### Run the Training Script

Execute the training process using the provided script:

```bash
python train.py --data_pq_file path/to/hmer_train.parquet \
                --dictionary_dir path/to/dictionary.txt \
                --save_tokenizer_dir path/to/custom_tokenizer.json \
                --checkpoint_dir path/to/save/checkpoints \
                --num_epochs 30 \
                --batch_size 8 \
                --learning_rate 3e-4 \
                --weight_decay 1e-4
```

### Inference

Ensure you have the test images.

```bash
python predict.py --checkpoint path/to/best_model.pth \
                  --tokenizer path/to/custom_tokenizer.json \
                  --test_folder path/to/test/imgs/ \
                  --output_file path/to/results/test_results_densenet.txt
```