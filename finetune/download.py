from huggingface_hub import snapshot_download


model_id = 'Qwen/Qwen2.5-VL-3B-Instruct'
save_dir = './model/base/Qwen2.5-VL-3B-Instruct/'
# snapshot_download(repo_id='Qwen/Qwen2.5-VL-3B-Instruct-AWQ', local_dir='./vl3b/', local_dir_use_symlinks=False)
snapshot_download(repo_id=model_id, local_dir=save_dir, local_dir_use_symlinks=False)