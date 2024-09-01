# Define a type for list items
from typing import TypedDict, List, Dict


class ExtractedItemHtml(TypedDict):
    type: str
    length: int


class ListItem(ExtractedItemHtml):
    children: List[str]


# Define a type for table items
class ContentItem(ExtractedItemHtml):
    content: str


# Define a type for table items
class TableItem(ContentItem):
    pass


class MathItem(ContentItem):
    pass


UUIDExtractedItemDict = Dict[str, ExtractedItemHtml]
