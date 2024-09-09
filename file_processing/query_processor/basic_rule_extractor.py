# add env vars before loading modules dependent on them (e.g. LLM API)
import logging
from enum import Enum
from typing import Optional, Any, List, Union

from langchain.chains.query_constructor.prompt import SONG_DATA_SOURCE, EXAMPLE_PROMPT
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers.json import parse_and_check_json_markdown
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.structured_query import StructuredQuery, Comparison, Operator, Comparator, FilterDirective, \
    Operation, Visitor

from file_processing.document_processor.llm_chat_support import chat_model

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    _format_attribute_info,
)
from langchain.chains.query_constructor.schema import AttributeInfo


class ExtendedComparator(str, Enum):
    """Extended comparator with negated forms."""
    NOT_CONTAIN = "not_contain"
    NOT_LIKE = "not_like"


class ExtendedComparison(Comparison):
    """Comparison to a value."""

    comparator: Union[Comparator, ExtendedComparator]
    attribute: str
    value: Any

    def __init__(
            self, comparator: Union[Comparator, ExtendedComparator], attribute: str, value: Any, **kwargs: Any
    ) -> None:
        super().__init__(
            comparator=comparator, attribute=attribute, value=value, **kwargs
        )


class NotToNegatedComparatorVisitor(Visitor):
    def visit_structured_query(self, structured_query: StructuredQuery) -> Any:
        pass

    def _validate_func(self, func: Union[Operator, Comparator]) -> None:
        pass

    negated_comparators = {
        Comparator.EQ: Comparator.NE,
        Comparator.NE: Comparator.EQ,
        Comparator.GT: Comparator.LTE,
        Comparator.GTE: Comparator.LT,
        Comparator.LT: Comparator.GTE,
        Comparator.LTE: Comparator.GT,
        Comparator.CONTAIN: ExtendedComparator.NOT_CONTAIN,
        Comparator.LIKE: ExtendedComparator.NOT_LIKE,
        Comparator.IN: Comparator.NIN,
        Comparator.NIN: Comparator.IN,
    }

    def visit_comparison(self, comparison: Comparison) -> FilterDirective:
        return comparison

    def visit_operation(self, operation: Operation) -> FilterDirective:
        if operation.operator == Operator.NOT and len(operation.arguments) == 1:
            argument = operation.arguments[0]
            if isinstance(argument, Comparison):
                negated_comparator = self.negated_comparators.get(argument.comparator)
                if negated_comparator:
                    return ExtendedComparison(
                        comparator=negated_comparator,
                        attribute=argument.attribute,
                        value=argument.value,
                    )
        # Recursively apply to arguments
        new_arguments = [arg.accept(self) for arg in operation.arguments]
        return Operation(operator=operation.operator, arguments=new_arguments)


class ComparisonCollectorVisitor(Visitor):
    def visit_structured_query(self, structured_query: StructuredQuery) -> Any:
        pass

    def __init__(self):
        self.comparison_list: List[Union[Comparison, ExtendedComparison]] = []

    def _validate_func(self, func: Union[Operator, Comparator]) -> None:
        pass

    def visit_comparison(self, comparison: Comparison) -> None:
        self.comparison_list.append(comparison)

    def visit_extended_comparison(self, comparison: ExtendedComparison) -> None:
        self.comparison_list.append(comparison)

    def visit_operation(self, operation: Operation) -> None:
        for argument in operation.arguments:
            argument.accept(self)


class StructuredQueryWithFailReason(StructuredQuery):
    fail_reason: Optional[str]
    """Reason to fail to extract filter string."""

    def __init__(
            self,
            query: str,
            filter: Optional[FilterDirective],
            limit: Optional[int] = None,
            fail_reason: Optional[str] = None,
            **kwargs: Any,
    ):
        super().__init__(query=query, filter=filter, limit=limit, fail_reason=fail_reason, **kwargs)

    def to_json(self):
        if self.filter is None:
            return {
                "query": self.query,
                "filter": [],
                "reason_for_no_filter": self.fail_reason,
            }
        visitor = NotToNegatedComparatorVisitor()
        # convert not to negated - e.g. not(like(smth)) -> not_like(smth)
        result = self.filter.accept(visitor)

        # add all filters to a list
        comparator_visitor = ComparisonCollectorVisitor()
        result.accept(comparator_visitor)

        placeholder_value = "__________"

        return {
            "query": self.query.replace(placeholder_value, ".") if self.query else None,
            "filter": [{
                "attribute": comparison.attribute.replace(placeholder_value, ".")
                if comparison.attribute else None,
                "comparator": f"{comparison.comparator}",
                "value": comparison.value.replace(placeholder_value, ".")
                if comparison.value else None,
            } for comparison in comparator_visitor.comparison_list],
            "reason_for_no_filter": self.fail_reason.replace(placeholder_value, ".")
            if self.fail_reason else None,
        }


class StructuredQueryOutputParserWithFailReason(StructuredQueryOutputParser,
                                                BaseOutputParser[StructuredQueryWithFailReason]):
    """Output parser that parses a structured query."""

    def parse(self, text: str) -> StructuredQueryWithFailReason:
        structured_query_res = super().parse(text.replace(".", "__________"))
        if structured_query_res.filter is None:
            expected_keys = ["reason_for_no_filter"]
            parsed = parse_and_check_json_markdown(text, expected_keys)
            print(f"Failed to parse filter because: {parsed['reason_for_no_filter']}")
            return StructuredQueryWithFailReason(
                query=structured_query_res.query,
                filter=structured_query_res.filter,
                limit=structured_query_res.limit,
                fail_reason=parsed["reason_for_no_filter"],
            )
        return StructuredQueryWithFailReason(
            query=structured_query_res.query,
            filter=structured_query_res.filter,
            limit=structured_query_res.limit,
            fail_reason=None,
        )


MEGA_PROMPT = """\
### Goal:
Format the user’s query using the schema below, always starting the `filter` with `and`.
Note: Use sample values for context only — do not use them directly in the response.


### Request Schema:
Return a markdown code snippet in the following JSON format:

```json
{
    "query": "string \ text to match document content",
    "filter": "string \ logical conditions starting with 'and'",
    "reason_for_no_filter": "string \ explanation if no filter is applied"
}
```

### Key Rules:
- **Query**: Only contains text expected to match document content.
- **Filter**:
  - Must **always** start with `and`.
  - Use only the allowed operators: `and`, `not`.
  - Comparisons are in this format: `comp(attr, val)`:
    - `comp`: (`eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `contain`, `in`, `nin`)
    - `attr`: Attribute name from the data source.
    - `val`: The comparison value.
- **No Filter**: Return `"NO_FILTER"` if:
  - The query needs unsupported operators (like `or`).
  - The query is too broad or unrelated to data source attributes.

### Important:
- **Do not use sample values directly**. They are provided for understanding how attributes and values may look, but the response should be based on the actual user query and data source.
  
---

### Example 1

**Data Source**:

```json
{{
    "content": "Lyrics of a song",
    "attributes": {{
        "artist": {{
            "type": "string",
            "description": "Here are some sample values for artist: John Lennon, Freddy Mercury, Bruce Springsteen"
        }},
        "length": {{
            "type": "integer",
            "description": "Length of the song in seconds"
        }},
        "genre": {{
            "type": "string",
            "description": "The song genre, one of \"pop\", \"rock\" or \"rap\""
        }}
    }}
}}
```

**User Query**: Songs by Taylor Swift about teenage romance under 3 minutes in pop.

**Structured Request**:

```json
{
    "query": "teenage love",
    "filter": "and(eq(\"artist\", \"Taylor Swift\"), lt(\"length\", 180), eq(\"genre\", \"pop\"))"
}
```

---

### Example 2

**Data Source**:
```json
{{
    "content": "Lyrics of a song",
    "attributes": {{
        "artist": {{
            "type": "string",
            "description": "Name of the song artist"
        }},
        "length": {{
            "type": "integer",
            "description": "Length of the song in seconds"
        }},
        "genre": {{
            "type": "string",
            "description": "The song genre, one of \"pop\", \"rock\" or \"rap\""
        }}
    }}
}}
```

**User Query**: Songs by Taylor Swift or Katy Perry about teenage romance under 3 minutes in pop.

**Structured Request**:

```json
{
    "query": "",
    "filter": "NO_FILTER",
    "reason_for_no_filter": "OR operator is not supported"
}
```

---

### Example 3
**Data Source**:
```json
{{
    "content": "Lyrics of a song",
    "attributes": {{
        "artist": {{
            "type": "string",
            "description": "Name of the song artist"
        }},
        "length": {{
            "type": "integer",
            "description": "Length of the song in seconds"
        }},
        "genre": {{
            "type": "string",
            "description": "The song genre, one of \"pop\", \"rock\" or \"rap\""
        }}
    }}
}}
```

**User Query**: Songs not on Spotify.

**Structured Request**:

```json
{
    "query": "",
    "filter": "NO_FILTER",
    "reason_for_no_filter": "Query is too broad"
}
```
"""

DEFAULT_SUFFIX = """\

---

### Actual Request
**Data Source**:
```json
{{{{
    "content": "{content}",
    "attributes": {attributes}
}}}}
```

**User Query**: {query}

**Structured Request**:
"""

allowed_comparators = [
    Comparator.EQ,
    Comparator.NE,
    Comparator.GT,
    Comparator.GTE,
    Comparator.LT,
    Comparator.LTE,
    Comparator.CONTAIN,
    Comparator.IN,
    Comparator.NIN
]
output_parser = StructuredQueryOutputParserWithFailReason.from_components(
    allowed_operators=[Operator.AND, Operator.NOT],
    allowed_comparators=allowed_comparators,
    fix_invalid=True,
)


def query_to_structured_filter(unstructured_query: str,
                               document_content_description: str,
                               metadata_field_info: List[AttributeInfo]) -> dict:
    attribute_str = _format_attribute_info(metadata_field_info)

    suffix = DEFAULT_SUFFIX.format(
        content=document_content_description,
        attributes=attribute_str,
        query=unstructured_query
    ).replace("{{", "{").replace("}}", "}")

    json_llm = chat_model.bind(response_format={"type": "json_object"})
    query_constructor = json_llm | output_parser
    res = query_constructor.invoke(MEGA_PROMPT + suffix)
    js = res.to_json()
    return js


if __name__ == "__main__":
    document_content_description = "Brief summary of a movie"
    metadata_field_info = [
        AttributeInfo(
            name="genre",
            description="The genre of the movie. One of ['science fiction', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated']",
            type="string",
        ),
        AttributeInfo(
            name="year",
            description="The year the movie was released",
            type="integer",
        ),
        AttributeInfo(
            name="director",
            description="The name of the movie director",
            type="string",
        ),
        AttributeInfo(
            name="rating", description="A 1-10 rating for the movie", type="float"
        ),
    ]
    res = query_to_structured_filter(
        "What are some movies from the 90's not directed by Luc Besson and the director name shouldn't contain Angle + in the sci-fi genre with a rating of at least 5?",
        document_content_description,
        metadata_field_info
    )
    print("res:", res)
