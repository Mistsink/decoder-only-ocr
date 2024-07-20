from fontTools.ttLib import TTFont
from PIL import ImageFont

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='fontTools.ttLib.tables._h_e_a_d')


_Fonts = {}
_ImageFonts = {}


def load_font(font_path) -> TTFont:
    if font_path not in _Fonts:
        _Fonts[font_path] = TTFont(font_path)
        # print(f"Loading font {font_path}")
    return _Fonts[font_path]


def load_image_font(font_path, font_size) -> ImageFont:
    if (font_path, font_size) not in _ImageFonts:
        _ImageFonts[(font_path, font_size)] = ImageFont.truetype(font_path, font_size)
        # print(f"Loading image font {font_path}")
    return _ImageFonts[(font_path, font_size)]


def has_char(font_file: str, char: str, depth: int = 0) -> bool:
    try:
        if font_file.split("/")[-1].startswith("LXGWWenKai"):
            return True

        font = load_font(font_file)
        uni_name = None
        for table in font["cmap"].tables:
            if ord(char) in table.cmap.keys():
                uni_name = table.cmap[ord(char)]

        if not uni_name:
            return False

        if "glyf" in font:
            glyf = font["glyf"]
            if uni_name in glyf and glyf[uni_name].numberOfContours > 0:
                return True
        elif "CFF " in font:
            cff = font["CFF "].cff
            if uni_name in cff.topDictIndex[0].CharStrings:
                image_f = load_image_font(font_file, 42)
                mask = image_f.getmask(char)
                if mask.size[0] > 0 and mask.size[1] > 0:
                    return True
        return False
    except Exception as e:
        if depth >= 20:
            return False
        return has_char(font_file, char, depth + 1)
