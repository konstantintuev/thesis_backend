class PdfToMdPageInfo:
    original_page_index: int
    raw_md_content: str
    fixed_md_content: str
    fix_log: str
    page_screenshot_path: str

    def __init__(self, original_page_index: int, raw_md_content: str, fixed_md_content: str, fix_log: str,
                 page_screenshot_path: str):
        super().__init__()
        self.original_page_index = original_page_index
        self.raw_md_content = raw_md_content
        self.fixed_md_content = fixed_md_content
        self.fix_log = fix_log
        self.page_screenshot_path = page_screenshot_path

    def get_best_text_content(self) -> str:
        # fixed_md_content is a processed version of raw_md_content,
        #   if we have it, we return it, else the raw_md_content
        if len(self.fixed_md_content) > 0:
            return self.fixed_md_content
        else:
            return self.raw_md_content


class PdfToMdDocument(list[PdfToMdPageInfo]):
    def get_best_text_content(self) -> str:
        return "\n\n".join([it.get_best_text_content() for it in self])

