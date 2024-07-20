from PIL import Image
import torch
from transformers import XGLMTokenizer, GenerationConfig

from config.config import cfg_from_yaml
from model.xglm import XGLMPatchForCausalLM
from dataset.transform import get_transform


cfg = cfg_from_yaml("config/debug.yaml")
tokenizer: XGLMTokenizer = XGLMTokenizer.from_pretrained(cfg.model, cache_dir="weight")
model = XGLMPatchForCausalLM.from_pretrained(
    cfg.model, patch_config=cfg, cache_dir="weight"
)
model.config.sep_token_id = tokenizer.sep_token_id

transform = get_transform(cfg.image_size)

img = Image.open("data/en.png").convert("RGB")
img = transform(img)
img = img.unsqueeze(0)  # add batch dimension

label_text = "torch.mean(prob).item()"
inputs = tokenizer(label_text, return_tensors="pt")  # 自动加上了 EOS/SEP token

model.train()

# output = model(
#     pixel_values=img, input_ids=inputs["input_ids"], labels=inputs["input_ids"]
# )
model.eval()
out = model.generate(
    pixel_values=img,
    max_length=30,
    num_beams=5,
    early_stopping=True,
)

t = tokenizer.decode(out[0], skip_special_tokens=True)

print(t)