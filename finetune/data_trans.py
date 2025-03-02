import os
import pickle
import cv2
import json
import pandas as pd

" Extract function "
def extract_img(pkl_file: str, save_dir: str=None):
    """
    Extracts images from a pickle file and saves them as image files in the specified directory.
    
    Parameters:
        pkl_file (str): Path to the pickle file containing image data.
        output_dir (str): Directory where the extracted images will be saved.
    """
    # Load the pickle file
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    # Ensure the output directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Iterate through the dictionary items
    for image_name, image_array in data.items():
        # Construct the full path for saving the image
        output_path = os.path.join(save_dir, image_name + '.png')
        
        # Save the image using OpenCV
        cv2.imwrite(output_path, image_array)
        print(f"Saved image: {output_path}")


extract_img(pkl_file='/root/autodl-tmp/crohme/2019/images.pkl', save_dir='../dataset/ft_data/')

def build_instruct_dataset(cap_text):
    with open(cap_text, 'r') as f:
        ext_pairs = f.readlines()
    pairs = [(title, cap) for ext in ext_pairs for title, cap in [ext.strip().split('\t')]]
    # print(pairs[:5], len(pairs))

    def add_prompt(idx, title, cap):
        MESSAGE = [
            {
                'id': f"pair-{idx + 1}",
                'conversation': [
                    {
                        "role": "user",
                        "url": '/root/autodl-tmp/HOCR/dataset/ft_data/' + title + '.png'
                    },
                    {
                        "role": "assistant",
                        "caption": '$$\t' + cap + '\t$$'
                    }
                ]
            }
        ]
        
        return { "message": MESSAGE }

    cvt_dataset = [add_prompt(idx, tlt, cap) for idx, (tlt, cap) in enumerate(pairs)]
    
    with open('ft_data.json', 'w', encoding='utf-8') as f:
        json.dump(cvt_dataset, f, ensure_ascii=False, indent=2)
    

# build_instruct_dataset(cap_text='./crohme/2019/caption.txt')
# print(data[:1], len(data))