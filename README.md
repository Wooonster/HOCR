# Handwriting OCR project

### Model architecture

ViT + Transformer

### Datasets and preprocessing

[CROHME](https://disk.pku.edu.cn/anyshare/en-us/link/AAF10CCC4D539543F68847A9010C607139?_tb=none&expires_at=1970-01-01T08%3A00%3A00%2B08%3A00&item_type=&password_required=false&title=HMER%20Dataset&type=anonymous).

To download and unzip:

```bash
wget https://disk.pku.edu.cn/anyshare/en-us/link/AAF10CCC4D539543F68847A9010C607139?_tb=none&expires_at=1970-01-01T08%3A00%3A00%2B08%3A00&item_type=&password_required=false&title=HMER%20Dataset&type=anonymous

unzip -d 'saved_zip'
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
```json
'train_31988.jpg': array([[141, 141, 143, ..., 152, 152, 152],
       [141, 141, 143, ..., 152, 152, 152],
       [144, 144, 143, ..., 153, 153, 153],
       ...,
       [144, 144, 144, ..., 149, 149, 149],
       [145, 145, 144, ..., 149, 149, 149],
       [145, 145, 144, ..., 149, 149, 149]], dtype=uint8)}
```

use Python package `pickle` to load or extract.


### init

`pip install -r requirements.txt`# HOCR
