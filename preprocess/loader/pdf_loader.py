"""Loader that loads image files."""
import os
from typing import List
import tempfile

import fitz
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from paddleocr import PaddleOCR
from langchain.schema import Document
import requests
from langchain.document_loaders.base import BaseLoader


class UnstructuredPaddlePDFLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load image files, such as PNGs and JPGs."""

    def _get_elements(self) -> List:
        def pdf_ocr_txt(filepath, dir_path="tmp_files"):
            full_dir_path = os.path.join(os.path.dirname(filepath), dir_path)
            if not os.path.exists(full_dir_path):
                os.makedirs(full_dir_path)
            filename = os.path.split(filepath)[-1]
            ocr = PaddleOCR(lang="ch", use_gpu=False, show_log=False)
            doc = fitz.open(filepath)
            txt_file_path = os.path.join(full_dir_path, "%s.txt" % (filename))
            img_name = os.path.join(full_dir_path, ".tmp.png")
            with open(txt_file_path, "w", encoding="utf-8") as fout:
                for i in range(doc.page_count):
                    page = doc[i]
                    text = page.get_text("")
                    fout.write(text)
                    fout.write("\n")

                    img_list = page.get_images()
                    for img in img_list:
                        pix = fitz.Pixmap(doc, img[0])

                        pix.save(img_name)

                        result = ocr.ocr(img_name)
                        ocr_result = [i[1][0] for line in result for i in line]
                        fout.write("\n".join(ocr_result))
            os.remove(img_name)
            return txt_file_path

        txt_file_path = pdf_ocr_txt(self.file_path)
        from unstructured.partition.text import partition_text

        return partition_text(filename=txt_file_path, **self.unstructured_kwargs)


class PdfLinkLoader(BaseLoader):
    """
    Load PDF from a link
    """

    def __init__(self, link: str, *args, **kwargs):
        if not self.valid_url(link):
            raise ValueError(f"Invalid url: {link}")
        self.link = link

    def load(self) -> List[Document]:
        with tempfile.NamedTemporaryFile() as f:
            f.write(requests.get(self.link).content)
            f.seek(0)
            loader = UnstructuredPaddlePDFLoader(f.name, mode="elements")
            return loader.load()

    @staticmethod
    def valid_url(url):
        return url.startswith("http://") or url.startswith("https://")
