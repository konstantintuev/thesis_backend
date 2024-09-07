import logging
import os
import tempfile
import time
import uuid
from threading import Lock

import os
import re
from typing import List, Tuple, Optional, Dict
import logging

from file_processing.document_processor.pdf_parsers.pdf_2_md_types import PdfToMdDocument, PdfToMdPageInfo
from file_processing.storage_manager import global_temp_dir

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import fitz  # PyMuPDF
import shapely.geometry as sg
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity
import concurrent.futures

from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader

from file_processing.document_processor.pdf_utils import split_pdf


class AzureDocIntelTPS:
    def __init__(self):
        self.tps = int(os.environ.get("AZURE_DOC_INTEL_MAX_TPS", "1"))
        self.interval = 1 / self.tps
        self.last_call = time.time() - self.interval
        self.lock = Lock()
        self.temp_dir = global_temp_dir

    def _wait_if_needed(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_call = time.time()

    """Mostly from gptpdf -->"""

    def _is_near(self, rect1: BaseGeometry, rect2: BaseGeometry, distance: float = 20) -> bool:
        """
        Check if two rectangles are near each other if the distance between them is less than the target.
        """
        return rect1.buffer(0.1).distance(rect2.buffer(0.1)) < distance

    def _is_horizontal_near(self, rect1: BaseGeometry, rect2: BaseGeometry, distance: float = 100) -> bool:
        """
        Check if two rectangles are near horizontally if one of them is a horizontal line.
        """
        result = False
        if abs(rect1.bounds[3] - rect1.bounds[1]) < 0.1 or abs(rect2.bounds[3] - rect2.bounds[1]) < 0.1:
            if abs(rect1.bounds[0] - rect2.bounds[0]) < 0.1 and abs(rect1.bounds[2] - rect2.bounds[2]) < 0.1:
                result = abs(rect1.bounds[3] - rect2.bounds[3]) < distance
        return result

    def _union_rects(self, rect1: BaseGeometry, rect2: BaseGeometry) -> BaseGeometry:
        """
        Union two rectangles.
        """
        return sg.box(*(rect1.union(rect2).bounds))

    def _merge_rects(self, rect_list: List[BaseGeometry], distance: float = 20,
                     horizontal_distance: Optional[float] = None) -> \
            List[BaseGeometry]:
        """
        Merge rectangles in the list if the distance between them is less than the target.
        """
        merged = True
        while merged:
            merged = False
            new_rect_list = []
            while rect_list:
                rect = rect_list.pop(0)
                for other_rect in rect_list:
                    if self._is_near(rect, other_rect, distance) or (
                            horizontal_distance and self._is_horizontal_near(rect, other_rect, horizontal_distance)):
                        rect = self._union_rects(rect, other_rect)
                        rect_list.remove(other_rect)
                        merged = True
                new_rect_list.append(rect)
            rect_list = new_rect_list
        return rect_list

    def _adsorb_rects_to_rects(self, source_rects: List[BaseGeometry], target_rects: List[BaseGeometry],
                               distance: float = 10) -> \
            Tuple[List[BaseGeometry], List[BaseGeometry]]:
        """
        Adsorb a set of rectangles to another set of rectangles.
        """
        new_source_rects = []
        for text_area_rect in source_rects:
            adsorbed = False
            for index, rect in enumerate(target_rects):
                if self._is_near(text_area_rect, rect, distance):
                    rect = self._union_rects(text_area_rect, rect)
                    target_rects[index] = rect
                    adsorbed = True
                    break
            if not adsorbed:
                new_source_rects.append(text_area_rect)
        return new_source_rects, target_rects

    def _parse_rects(self, page: fitz.Page) -> List[Tuple[float, float, float, float]]:
        """
        Parse drawings in the page and merge adjacent rectangles.
        """

        # 提取画的内容
        drawings = page.get_drawings()

        # 忽略掉长度小于30的水平直线
        is_short_line = lambda x: abs(x['rect'][3] - x['rect'][1]) < 1 and abs(x['rect'][2] - x['rect'][0]) < 30
        drawings = [drawing for drawing in drawings if not is_short_line(drawing)]

        # 转换为shapely的矩形
        rect_list = [sg.box(*drawing['rect']) for drawing in drawings]

        # 提取图片区域
        images = page.get_image_info()
        image_rects = [sg.box(*image['bbox']) for image in images]

        # 合并drawings和images
        rect_list += image_rects

        merged_rects = self._merge_rects(rect_list, distance=10, horizontal_distance=100)
        merged_rects = [rect for rect in merged_rects if explain_validity(rect) == 'Valid Geometry']

        # 将大文本区域和小文本区域分开处理: 大文本相小合并，小文本靠近合并
        is_large_content = lambda x: (len(x[4]) / max(1, len(x[4].split('\n')))) > 5
        small_text_area_rects = [sg.box(*x[:4]) for x in page.get_text('blocks') if not is_large_content(x)]
        large_text_area_rects = [sg.box(*x[:4]) for x in page.get_text('blocks') if is_large_content(x)]
        _, merged_rects = self._adsorb_rects_to_rects(large_text_area_rects, merged_rects, distance=0.1)  # 完全相交
        _, merged_rects = self._adsorb_rects_to_rects(small_text_area_rects, merged_rects, distance=5)  # 靠近

        # 再次自身合并
        merged_rects = self._merge_rects(merged_rects, distance=10)

        # 过滤比较小的矩形
        merged_rects = [rect for rect in merged_rects if
                        rect.bounds[2] - rect.bounds[0] > 20 and rect.bounds[3] - rect.bounds[1] > 20]

        return [rect.bounds for rect in merged_rects]

    def _parse_pdf_to_images(self, pdf_path: str, output_dir: str = './') -> List[Tuple[str, List[str]]]:
        """
        Parse PDF to images and save to output_dir.
        """
        # 打开PDF文件
        pdf_document = fitz.open(pdf_path)
        image_infos = []

        fontname = "helv"
        fontsize = 10
        font = fitz.Font(fontname=fontname)

        for page_index, page in enumerate(pdf_document):
            logging.info(f'parse page: {page_index}')
            rect_images = []
            rects = self._parse_rects(page)
            for index, rect in enumerate(rects):
                # My own better logic for text boxes below (above and length aware
                #   instead of below and not length aware)
                fitz_rect = fitz.Rect(rect)
                # 保存页面为图片
                pix = page.get_pixmap(clip=fitz_rect, matrix=fitz.Matrix(4, 4))
                name = f'{page_index}_{index}.png'
                name_ui = f'<img src="{page_index}_{index}.png"/>'
                pix.save(os.path.join(output_dir, name))
                rect_images.append(name)
                # # 在页面上绘制红色矩形
                big_fitz_rect = fitz.Rect(fitz_rect.x0 - 1, fitz_rect.y0 - 1, fitz_rect.x1 + 1, fitz_rect.y1 + 1)
                # 空心矩形
                page.draw_rect(big_fitz_rect, color=(1, 0, 0), width=1)
                # 画矩形区域(实心)
                # page.draw_rect(big_fitz_rect, color=(1, 0, 0), fill=(1, 0, 0))
                # 在矩形内的左上角写上矩形的索引name，添加一些偏移量
                # at least 0, but less than page height
                text_y_on_top = min(max(fitz_rect.y0 - fontsize/2 - 1, 0), page.rect.height - fontsize/2 - 2)
                text_width = font.text_length(name_ui, fontsize=fontsize)
                text_x_norm = min(fitz_rect.x0 + 2, page.rect.width - text_width)
                # x0, y0, x1, y1c
                text_rect = fitz.Rect(text_x_norm - 2, text_y_on_top - fontsize/2 - 3,
                                      text_x_norm + text_width + 2, text_y_on_top + fontsize/2 - 1)
                # 绘制白色背景矩形
                page.draw_rect(text_rect, color=None, fill=(1, 1, 1), fill_opacity=0.8)
                # 插入带有白色背景的文字
                page.insert_text((text_x_norm, text_y_on_top), name_ui, fontsize=fontsize, color=(1, 0, 0),
                                 fontname=fontname)
            page_image_with_rects = page.get_pixmap(matrix=fitz.Matrix(3, 3))
            page_image = os.path.join(output_dir, f'{page_index}.png')
            page_image_with_rects.save(page_image)
            image_infos.append((page_image, rect_images))

        pdf_document.close()
        return image_infos

    def _gpt_parse_images(
            self,
            image_infos: List[Tuple[str, List[str]]],
            output_dir: str = './',
            verbose: bool = False,
            gpt_worker: int = 1,
            **args
    ) -> str:
        """
        Parse images to markdown content.
        """

        def _process_page(index: int, image_info: Tuple[str, List[str]]) -> Tuple[int, str]:
            self._wait_if_needed()
            logging.info(f'gpt parse page: {index}')
            page_image, rect_images = image_info
            content = ""

            analysis_features = ["ocrHighResolution", "formulas"]
            # Create a temp dir into which we split the pdf - this way we honor the max pages requirement
            loader = AzureAIDocumentIntelligenceLoader(
                api_endpoint=os.environ.get("AZURE_DOC_INTEL_ENDPOINT"),
                api_key=os.environ.get("AZURE_DOC_INTEL_API_KEY"),
                file_path=page_image,
                api_model="prebuilt-layout",
                mode="markdown",
                analysis_features=analysis_features,
            )

            documents = loader.load()
            for document in documents:
                content += document.page_content + "\n\n"
            return index, content

        contents = [None] * len(image_infos)
        with concurrent.futures.ThreadPoolExecutor(max_workers=gpt_worker) as executor:
            futures = [executor.submit(_process_page, index, image_info) for index, image_info in
                       enumerate(image_infos)]
            for future in concurrent.futures.as_completed(futures):
                index, content = future.result()

                contents[index] = content

        output_path = os.path.join(output_dir, 'output.md')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(contents))

        return '\n\n'.join(contents)

    def pdf_to_md_azure_doc_intel_ocr(
            self,
            pdf_path: str,
            output_dir: str = './',
            verbose: bool = False,
            **args
    ) -> Tuple[str, List[str]]:
        """
        Parse a PDF file to a markdown file.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_infos = self._parse_pdf_to_images(pdf_path, output_dir=output_dir)
        content = self._gpt_parse_images(
            image_infos=image_infos,
            output_dir=output_dir,
            verbose=verbose,
            **args
        )

        all_rect_images = []
        # remove all rect images
        if not verbose:
            for page_image, rect_images in image_infos:
                if os.path.exists(page_image):
                    os.remove(page_image)
                all_rect_images.extend(rect_images)
        return content, all_rect_images

    """<-- Mostly from gptpdf"""

    def pdf_to_md_azure_doc_intel_pdfs(self, pdf_filepath: str, output_dir: str) -> PdfToMdDocument:
        self._wait_if_needed()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        split_pdf_paths = split_pdf(pdf_filepath,
                                    output_dir,
                                    2,
                                    True)
        pages_out = PdfToMdDocument()
        analysis_features = ["ocrHighResolution", "formulas"]
        for split_pdf_path in split_pdf_paths:
            # Create a temp dir into which we split the pdf - this way we honor the max pages requirement
            loader = AzureAIDocumentIntelligenceLoader(
                api_endpoint=os.environ.get("AZURE_DOC_INTEL_ENDPOINT"),
                api_key=os.environ.get("AZURE_DOC_INTEL_API_KEY"),
                file_path=split_pdf_path.split_pdf_path,
                api_model="prebuilt-layout",
                mode="markdown",
                analysis_features=analysis_features,
            )

            raw_page_md_content = ""
            documents = loader.load()
            for document in documents:
                raw_page_md_content += document.page_content + "\n\n"
            pages_out.append(PdfToMdPageInfo(
                split_pdf_path.from_original_start_page,
                raw_page_md_content,
                "",
                "",
                split_pdf_path.screenshots_per_page[0]
            ))
        return pages_out


azure_doc_intel_impl = AzureDocIntelTPS()


def pdf_to_md_azure_doc_intel(pdf_filepath: str, out_dir: str) -> PdfToMdDocument:
    res = azure_doc_intel_impl.pdf_to_md_azure_doc_intel_pdfs(pdf_filepath, out_dir)
    return res


if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.DEBUG)
    import argparse

    parser = argparse.ArgumentParser(description='PDF to MD with Azure Document Intelligence')
    parser.add_argument('input_pdf', type=str, help='Path to the input PDF file.')
    args = parser.parse_args()
    print(pdf_to_md_azure_doc_intel(args.input_pdf))