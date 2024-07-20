from typing import List
from PIL import Image
from torchvision import transforms as T


def pad_to_aspect_ratio(
    img: Image.Image,
    target_size: tuple[int, int],
    fill: tuple[int, int, int] = (255, 255, 255),
):
    # 获取目标宽高比
    target_ratio = target_size[0] / target_size[1]
    # 获取图像的实际宽高比
    img_ratio = img.width / img.height

    if img_ratio > target_ratio:
        new_width = target_size[0]
        new_height = int(new_width / img_ratio)
    else:
        new_height = target_size[1]
        new_width = int(new_height * img_ratio)

    # 缩放图像
    img = img.resize((new_width, new_height), Image.BICUBIC)

    # 创建新图像并将缩放后的图像粘贴进去
    new_img = Image.new("RGB", target_size, fill)
    new_img.paste(
        img, ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2)
    )

    return new_img


def get_transform(img_size: List[int]):
    transforms = []

    transforms.extend(
        [
            lambda img: pad_to_aspect_ratio(img, img_size, fill=(255, 255, 255)),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ]
    )
    return T.Compose(transforms)
