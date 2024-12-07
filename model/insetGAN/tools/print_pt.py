import torch

# latents.pt 파일 경로
latents_path = "/workspace/encoder4editing/weights/latents.pt"

# 파일 열기
latents = torch.load(latents_path)

# 내용 출력
if isinstance(latents, dict):
    print("Latents is a dictionary with keys:", latents.keys())
    for key, value in latents.items():
        print(f"Key: {key}, Type: {type(value)}, Shape: {value.shape if isinstance(value, torch.Tensor) else 'N/A'}")
elif isinstance(latents, torch.Tensor):
    print("Latents is a tensor with shape:", latents.shape)
else:
    print("Latents is of type:", type(latents))
