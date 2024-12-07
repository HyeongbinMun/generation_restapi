import pickle
import torch
import sys

sys.path.append('/workspace')
# .pkl 파일 경로
ckpt_path = '/workspace/networks/DeepFashion_1024x768.pkl'


def explore_pkl(ckpt_path):
    """Explore the structure of a .pkl file."""
    with open(ckpt_path, 'rb') as f:
        data = pickle.Unpickler(f).load()

    print(f"Keys in the .pkl file: {list(data.keys())}\n")

    for key, value in data.items():
        print(f"Key: {key}")
        print(f"Type: {type(value)}")
        if isinstance(value, torch.nn.Module):
            print("This is a PyTorch model.")
        elif isinstance(value, torch.Tensor):
            print(f"Tensor shape: {value.shape}")
        elif isinstance(value, dict):
            print(f"Dict keys: {list(value.keys())}")
        else:
            print(f"Value: {value}")
        print("-" * 50)


# 실행
explore_pkl(ckpt_path)