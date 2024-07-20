import os
import random
from typing import Dict, List, Tuple
from PIL import Image, ImageFont

from .font_support import has_char

from .data_generator import FakeTextDataGenerator
from .string_generator import create_strings_from_dict, create_strings_from_dict_index
from .util import create_line_index, find_all_newlines, load_fonts, load_dict


class DataGenerator:

    lang_fonts: Dict[str, List[str]]  # lang: [font1, font2, ...]
    lang_dict: Dict[str, List[str]]  # lang: [word1, word2, ...]

    def __init__(
        self,
        lang_dir: str,
        dict_dir: str,
        max_length: int,
        background_img_dir: str = None,
        max_sentence_len: int = 30,
    ) -> None:
        """
        lang_dir: lang1/(x.ttf, xx.otf), lang2/(x.ttf, xx.otf), ... (lang1, lang2, ... are language names
        dict_dir: lang1.txt, lang2.txt, ...
        """
        self._load_fonts(lang_dir)
        self._load_dict(dict_dir)
        self.max_length = max_length
        self.max_sentence_len = max_sentence_len
        self.background_img_dir = background_img_dir

    def _load_fonts(self, lang_dir: str) -> None:
        self.lang_fonts = {}
        for lang in os.listdir(lang_dir):
            if lang.startswith(".") and os.path.isfile(os.path.join(lang_dir, lang)):
                continue
            self.lang_fonts[lang] = load_fonts(os.path.join(lang_dir, lang))

    def _load_dict(self, dict_dir: str) -> None:
        """
        dict_dir: lang1.txt, lang2.txt, ...
        exclude files start with "merged"
        """
        self.lang_dict: Dict[str, Tuple[str, List[int]]] = {}
        for dict_file in os.listdir(dict_dir):
            if dict_file.startswith("merge") or dict_file.startswith("."):
                continue
            lang = dict_file.split(".")[0]
            if lang not in self.lang_fonts:
                continue

            file_path = os.path.join(dict_dir, dict_file)
            print(f'Loading {file_path}...')
            self.lang_dict[lang] = (file_path, create_line_index(file_path))
            print(f'Loaded {file_path}.')
        print(f"Loaded {len(self.lang_dict)} languages.")

    def generate(self, verbose: bool = False) -> Tuple[Image.Image, Image.Image, str]:
        """
        :return: (image, mask)
        """
        # background_type   0 -> 高斯模糊背景, 1 -> 纯色背景，2 -> 准晶体背景, x -> 自定义背景,
        background_type = random.choice([-4, -3, -2, -1, 0, 1, 2])
        # font_size
        MAX_FONT_SIZE = 70
        font_size = random.choice(range(10, MAX_FONT_SIZE + 1))

        # angle
        MAX_ANGLE = 30
        angle = random.choice(range(-MAX_ANGLE, MAX_ANGLE + 1))
        if random.random() < 0.8:  # 大部分情况不旋转
            angle = 0

        # image_width
        image_width = random.choice(range(128, 512 + 1))

        # text_color
        text_color = "#{:02x}{:02x}{:02x}".format(
            random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        )

        # stroke
        stroke_width = random.choice(range(0, 3))
        stroke_fill = "#{:02x}{:02x}{:02x}".format(
            random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        )

        # choose lang and font
        langs = list(self.lang_dict.keys())
        lang_num = 1 if random.random() < 0.6 else 2
        langs = random.choices(langs, k=lang_num)
        fonts = [random.choice(self.lang_fonts[lang]) for lang in langs]
        image_fonts = {
            font: ImageFont.truetype(font=font, size=font_size) for font in fonts
        }

        # generate text
        MAX_NUM_LINE_BREAK = 4
        MAX_NUM_SPACE = 4 * 8
        text_max_length = self.max_length - MAX_NUM_LINE_BREAK - MAX_NUM_SPACE
        texts: List[List[str]] = []
        for lang in langs:
            # + 1 是为了尽量让文字短一点，让多语言的文字更多
            max_length = text_max_length // (lang_num + 1)
            count = random.choice([1, 2])
            _texts = create_strings_from_dict_index(
                max_length,
                allow_variable=True,
                count=count,
                file_path=self.lang_dict[lang][0],
                index=self.lang_dict[lang][1],
            )
            texts.append(_texts)

        # refine texts
        # 1. total_length < text_max_length
        _total_length = 0
        final_texts = []
        for _texts in texts:
            cur_texts = []
            for text in _texts:
                if _total_length + len(text) > text_max_length:
                    _text = text[: text_max_length - _total_length]
                    cur_texts.append(_text)
                    _total_length += len(_text)
                    break
                else:
                    _total_length += len(text)
                    cur_texts.append(text)
            final_texts.append(cur_texts)
        texts = final_texts
        langs = langs[: len(texts)]
        fonts = fonts[: len(texts)]
        assert len(texts) == len(langs) == len(fonts)

        # 2. add line break and space
        # num_line_break = random.choice(range(MAX_NUM_LINE_BREAK + 1))
        # total_len = sum([len(text) for _texts in texts for text in _texts])
        # idxs_line_break = sorted(
        #     list(set(random.choices(range(total_len), k=num_line_break)))
        # )
        # texts = self.insert_str_into_text_list(texts, idxs_line_break, "\n")

        # total_len = sum([len(text) for _texts in texts for text in _texts])
        # num_space = random.choice(range(MAX_NUM_SPACE + 1))
        # idxs_space = sorted(list(set(random.choices(range(total_len), k=num_space))))
        # texts = self.insert_str_into_text_list(texts, idxs_space, " ")

        text_data: List[Tuple[str, str, str]] = []  # [(text, lang, font), ...]
        for i, _texts in enumerate(texts):
            text_data.extend([(text, langs[i], fonts[i]) for text in _texts])
        random.shuffle(text_data)

        # 3. refine line_break and space
        text_data = list(
            map(
                lambda x: (
                    self.refine_text(
                        x[0],
                        rm_pre_last_space=False,
                        lang=x[1],
                        font_file=x[2],
                        font=image_fonts[x[2]],
                    ),
                    x[1],
                    x[2],
                ),
                text_data,
            )
        )

        sentences: List[List[Tuple[str, str, str]]] = []
        # cur_sentence = []
        for text, lang, font in text_data:
            # line_break_idxs = find_all_newlines(text)
            # _pre_idx = 0
            # for idx in line_break_idxs:
            #     cur_sentence.append((text[_pre_idx:idx], lang, font))
            #     sentences.append(cur_sentence)
            #     cur_sentence = []
            #     _pre_idx = idx + 1
            # cur_sentence.append((text[_pre_idx:], lang, font))
            sentences.append([(text, lang, font)])

        # if cur_sentence:
        #     sentences.append(cur_sentence)

        sentences = refine_sentences_maxlen(sentences, self.max_sentence_len)

        sentence_text = []
        for sentence in sentences:
            cur_sentence_text = ""
            for text, _, _ in sentence:
                cur_sentence_text += text
            sentence_text.append(cur_sentence_text)
        target_text = "\n".join(sentence_text)
        target_text = self.refine_text(target_text, rm_pre_last_space=True)
        # 将全角空格替换成正常的空格
        target_text = target_text.replace("\u3000", " ")

        # direction
        # langs 只有汉子、日文，80%是竖排，20%是横排、其他语言都是横排
        only_cjk = all([lang in ["cn", "zh", "zh_cn", "zh_tw", "ja"] for lang in langs])
        orientation = 0  # 0 -> horizontal, 1 -> vertical
        if only_cjk:
            orientation = 1 if random.random() < 0.8 else 0

        # generate image
        image, mask = FakeTextDataGenerator.generate(
            index=0,
            text=target_text,
            sentences=sentences,
            out_dir=None,
            size=font_size,
            extension="jpg",
            skewing_angle=angle,
            random_skew=False,
            blur=0,
            random_blur=True,
            background_type=background_type,  # 0 -> 高斯模糊背景, 1 -> 纯色背景，2 -> 准晶体背景, x -> 自定义背景,
            distorsion_type=0,
            distorsion_orientation=0,
            name_format=0,
            width=image_width,
            alignment=1,
            text_color=text_color,
            orientation=orientation,
            space_width=1,
            character_spacing=0,
            margins=(5, 5, 5, 5),
            fit=False,
            output_mask=True,
            word_split=False,
            image_dir=self.background_img_dir,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
            image_mode="RGB",
        )
        if verbose and image and mask:
            if not os.path.exists("data/tmp"):
                os.makedirs("data/tmp")
            image.save("data/tmp/text.png")
            mask.save("data/tmp/mask.png")

        return image, mask, target_text

    @staticmethod
    def insert_str_into_text_list(
        texts: List[List[str]], idxs: List[int], insert_str: str
    ) -> List[List[str]]:
        if not idxs:
            return texts
        length = 0
        cur_idx = idxs.pop(0)
        for i in range(len(texts)):
            for j in range(len(texts[i])):
                while cur_idx is not None and length + len(texts[i][j]) >= cur_idx:
                    insert_idx = cur_idx - length
                    texts[i][j] = (
                        texts[i][j][:insert_idx] + insert_str + texts[i][j][insert_idx:]
                    )
                    if idxs:
                        cur_idx = idxs.pop(0)
                    else:
                        cur_idx = None
                if cur_idx is None:
                    return texts
                length += len(texts[i][j])

        return texts

    @staticmethod
    def refine_text(
        text,
        rm_pre_last_space=False,
        lang: str = None,
        font_file: str = None,
        font: ImageFont = None,
    ):
        force_up = False
        if font_file and font_file.split("/")[-1].startswith("UP_"):
            force_up = True

        if font:
            # 清理不支持的字符
            chars = set(text)
            for char in chars:
                if char == "\n" or char == " ":
                    continue
                if not has_char(font_file, char):
                    text = text.replace(char, "")

        # 删除连续的换行和空格
        # text = re.sub(r"\n+", "\n", text)  # 将多个换行符替换为单个换行符
        # text = re.sub(r" +", " ", text)  # 将多个空格替换为单个空格

        # # 确保换行前后没有空格
        # text = re.sub(r"\s*\n\s*", "\n", text)

        if force_up:
            # 将所有字母转为大写
            text = text.upper()

        # 移除文本开头和末尾的空格或换行
        if rm_pre_last_space:
            text = text.strip()

        return text


def refine_sentences_maxlen(
    sentences: List[List[Tuple[str, str, str]]], max_length: int
) -> List[List[Tuple[str, str, str]]]:
    # 修剪长句子
    _sentences = []
    for i in range(len(sentences)):
        _cur_sentence = []
        cur_sen_len = 0
        for j in range(len(sentences[i])):
            if cur_sen_len + len(sentences[i][j][0]) > max_length:
                _cur_sentence.append(
                    (
                        sentences[i][j][0][: max_length - cur_sen_len],
                        sentences[i][j][1],
                        sentences[i][j][2],
                    )
                )
                break
            else:
                _cur_sentence.append(sentences[i][j])
                cur_sen_len += len(sentences[i][j][0])

        if _cur_sentence:
            _sentences.append(_cur_sentence)
    sentences = _sentences

    # 清除为空的句子
    _sentences = []
    for i in range(len(sentences)):
        _cur_sentence = []
        for j in range(len(sentences[i])):
            if not sentences[i][j][0]:
                continue
            _cur_sentence.append(sentences[i][j])

        if _cur_sentence:
            _sentences.append(_cur_sentence)
    sentences = _sentences

    return sentences


def main(n: int = 3):
    g = DataGenerator(
        lang_dir="assets/font",
        dict_dir="assets/dict",
        background_img_dir="assets/background",
        max_length=128,
        max_sentence_len=10,
    )
    for _ in range(n):
        img, mask, label = g.generate(verbose=True)


if __name__ == "__main__":
    main()
