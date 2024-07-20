"""
Utility functions
"""

import os
import random
import re
import unicodedata
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_dict(path: str) -> List[str]:
    """Read the dictionary file and returns all words in it."""

    word_dict = []
    with open(
        path,
        "r",
        encoding="utf8",
        errors="ignore",
    ) as d:
        word_dict = [l for l in d.read().splitlines() if len(l) > 0]

    return word_dict


def load_fonts(dir: str) -> List[str]:
    """Load all fonts in the fonts directories"""

    font_file_ext = [".ttf", ".otf"]

    fonts = []
    for font in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, font)) and any(
            font.endswith(ext) for ext in font_file_ext
        ):
            fonts.append(os.path.join(dir, font))

    return fonts


def mask_to_bboxes(mask: List[Tuple[int, int, int, int]], tess: bool = False):
    """Process the mask and turns it into a list of AABB bounding boxes"""

    mask_arr = np.array(mask)

    bboxes = []

    i = 0
    space_thresh = 1
    while True:
        try:
            color_tuple = ((i + 1) // (255 * 255), (i + 1) // 255, (i + 1) % 255)
            letter = np.where(np.all(mask_arr == color_tuple, axis=-1))
            if space_thresh == 0 and letter:
                x1 = min(bboxes[-1][2] + 1, np.min(letter[1]) - 1)
                y1 = (
                    min(bboxes[-1][3] + 1, np.min(letter[0]) - 1)
                    if not tess
                    else min(
                        mask_arr.shape[0] - np.min(letter[0]) + 2, bboxes[-1][1] - 1
                    )
                )
                x2 = max(bboxes[-1][2] + 1, np.min(letter[1]) - 2)
                y2 = (
                    max(bboxes[-1][3] + 1, np.min(letter[0]) - 2)
                    if not tess
                    else max(
                        mask_arr.shape[0] - np.min(letter[0]) + 2, bboxes[-1][1] - 1
                    )
                )
                bboxes.append((x1, y1, x2, y2))
                space_thresh += 1
            bboxes.append(
                (
                    max(0, np.min(letter[1]) - 1),
                    (
                        max(0, np.min(letter[0]) - 1)
                        if not tess
                        else max(0, mask_arr.shape[0] - np.max(letter[0]) - 1)
                    ),
                    min(mask_arr.shape[1] - 1, np.max(letter[1]) + 1),
                    (
                        min(mask_arr.shape[0] - 1, np.max(letter[0]) + 1)
                        if not tess
                        else min(
                            mask_arr.shape[0] - 1,
                            mask_arr.shape[0] - np.min(letter[0]) + 1,
                        )
                    ),
                )
            )
            i += 1
        except Exception as ex:
            if space_thresh == 0:
                break
            space_thresh -= 1
            i += 1

    return bboxes


def draw_bounding_boxes(
    img: Image, bboxes: List[Tuple[int, int, int, int]], color: str = "green"
) -> None:
    d = ImageDraw.Draw(img)

    for bbox in bboxes:
        d.rectangle(bbox, outline=color)


def make_filename_valid(value: str, allow_unicode: bool = False) -> str:
    """
    Code adapted from: https://docs.djangoproject.com/en/4.0/_modules/django/utils/text/#slugify

    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value)

    # Image names will be shortened to avoid exceeding the max filename length
    return value[:200]


def get_text_width(image_font: ImageFont, text: str) -> int:
    """
    Get the width of a string when rendered with a given font
    """
    return round(image_font.getlength(text))


def get_text_height(image_font: ImageFont, text: str) -> int:
    """
    Get the height of a string when rendered with a given font
    """
    left, top, right, bottom = image_font.getbbox(text)
    return bottom


# Function to concatenate images horizontally
def get_concat_h(im1: Image.Image, im2: Image.Image):
    dst = Image.new("RGBA", (im1.width + im2.width, max(im1.height, im2.height)))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


# Function to concatenate images horizontally with im2 to the left
def get_concat_h_left(im1: Image.Image, im2: Image.Image):
    dst = Image.new("RGBA", (im1.width + im2.width, max(im1.height, im2.height)))
    dst.paste(im2, (0, 0))
    dst.paste(im1, (im2.width, 0))
    return dst


# Function to concatenate images vertically
def get_concat_v(im1: Image.Image, im2: Image.Image):
    dst = Image.new("RGBA", (max(im1.width, im2.width), im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


# Function to concatenate images vertically and center horizontally
def get_concat_v_center(im1: Image.Image, im2: Image.Image):
    # if im1.width < im2.width:
    #     im1, im2 = im2, im1

    # assert im1.width >= im2.width
    im1_larger = im1.width >= im2.width
    max_w = max(im1.width, im2.width)
    dst = Image.new("RGBA", (max_w, im1.height + im2.height))
    width_offset = abs(im1.width - im2.width) // 2
    if im1_larger:
        dst.paste(im1, (0, 0))
        dst.paste(im2, (width_offset, im1.height))
    else:
        dst.paste(im1, (width_offset, 0))
        dst.paste(im2, (0, im1.height))
    return dst
    # dst = Image.new("RGBA", (max(im1.width, im2.width), im1.height + im2.height))
    # width_offset = (dst.width - im2.width) // 2
    # dst.paste(im1, (0, 0))
    # dst.paste(im2, (width_offset, im1.height))

    # return dst


def find_all_newlines(text: str) -> List[int]:
    idx = text.find("\n")
    idxs = []
    while idx != -1:
        idxs.append(idx)
        idx = text.find("\n", idx + 1)

    return idxs


def create_line_index(file_path: str) -> List[int]:
    index: List[int] = []
    with open(file_path, "r", encoding="utf-8") as file:
        position = file.tell()
        line = file.readline()
        while line:
            index.append(position)
            position = file.tell()
            line = file.readline()
    return index


def read_random_line(file_path: str, index: List[int]) -> str:
    random_position = random.choice(index)
    with open(file_path, "r", encoding="utf-8") as file:
        file.seek(random_position)
        line = file.readline()
    return line.strip()
