# Define a type for list items
from typing import TypedDict, List, Dict


class ExtractedItemHtml(TypedDict):
    type: str
    length: int

class ListItem(ExtractedItemHtml):
    children: List[str]


# Define a type for table items
class TableItem(ExtractedItemHtml):
    content: str


UUIDExtractedItemDict = Dict[str, ExtractedItemHtml]