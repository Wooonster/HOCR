# HOCR: Handwriting OCR project

### Model architecture

#### ~~ViT~~ + Transformer

*Reason why ViT doesn't work well*: 

Domain differences in pre-training ViT models: ViT (Vision Transformer) is usually pre-trained on large-scale natural image datasets like ImageNet. Your dataset, on the other hand, is an image of handwritten mathematical formulas, which is very different from natural images in terms of feature distribution. A pre-trained ViT model may not be able to extract valid features, resulting in encoder output that is meaningless to the decoder.

Lack of low-level feature extraction: handwritten formula images have complex local features, such as stroke and symbol details. viT slices the image directly into patches and then performs a global self-attention mechanism, which may not be able to capture these critical local features.

#### Corrections / TODOs

Try implementing DenseNet instead, or combine DenseNet with ViT.


### Datasets and preprocessing

[CROHME](https://disk.pku.edu.cn/anyshare/en-us/link/AAF10CCC4D539543F68847A9010C607139?_tb=none&expires_at=1970-01-01T08%3A00%3A00%2B08%3A00&item_type=&password_required=false&title=HMER%20Dataset&type=anonymous).

**Notice:** This link contains two datasets, CROHME and HMER, download from this link and extract CROHME.zip from the zip.

To ~~download and~~ unzip:

~~wget https://disk.pku.edu.cn/anyshare/en-us/link/AAF10CCC4D539543F68847A9010C607139?_tb=none&expires_at=1970-01-01T08%3A00%3A00%2B08%3A00&item_type=&password_required=false&title=HMER%20Dataset&type=anonymous~~

```bash
unzip filename.zip -d /path/to/destination
```

#### Unzipped structure:

```bash
crohme % tree
.
├── 2014
│   ├── caption.txt
│   └── images.pkl
├── 2016
│   ├── caption.txt
│   └── images.pkl
├── 2019
│   ├── caption.txt
│   └── images.pkl
├── crohme_dictionary.txt
└── train
    ├── caption.txt
    └── images.pkl
```

#### image file type

The images are stored in the `.pkl` file as following structure:
```
'train_31988.jpg': 
    array([[141, 141, 143, ..., 152, 152, 152],
        [141, 141, 143, ..., 152, 152, 152],
        [144, 144, 143, ..., 153, 153, 153],
        ...,
        [144, 144, 144, ..., 149, 149, 149],
        [145, 145, 144, ..., 149, 149, 149],
        [145, 145, 144, ..., 149, 149, 149]], dtype=uint8)
```

use Python package `pickle` to load or extract.


### init

The versions of the packages listed in requirements.txt are not guaranteed to work.

`pip install -r requirements.txt`
