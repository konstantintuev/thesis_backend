# add env vars before loading modules dependent on them (e.g. LLM API)
import logging
from abc import ABC
from enum import Enum
from typing import Optional, Any, List, Union

from langchain.chains.query_constructor.prompt import SONG_DATA_SOURCE, EXAMPLE_PROMPT
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers.json import parse_and_check_json_markdown
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.structured_query import StructuredQuery, Comparison, Operator, Comparator, FilterDirective, \
    Operation, Visitor

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt, load_query_constructor_runnable, _format_attribute_info,
)
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from file_processing.embeddings import embeddings_model


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

        return {
            "query": self.query,
            "filter": [{
                "attribute": comparison.attribute,
                "comparator": f"{comparison.comparator}",
                "value": comparison.value,
            } for comparison in comparator_visitor.comparison_list],
            "reason_for_no_filter": self.fail_reason,
        }

class StructuredQueryOutputParserWithFailReason(StructuredQueryOutputParser,
                                                BaseOutputParser[StructuredQueryWithFailReason]):
    """Output parser that parses a structured query."""

    def parse(self, text: str) -> StructuredQueryWithFailReason:
        structured_query_res = super().parse(text)
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

DEFAULT_SCHEMA = """\
<< Structured Request Schema >>
When responding use a markdown code snippet with a JSON object formatted in the following schema:

```json
{{{{
    "query": "string \\ text string to compare to document contents",
    "filter": "string \\ logical condition statement for filtering documents"
}}}}
```

The query string should contain only text that is expected to match the contents of documents. Any conditions in the filter should not be mentioned in the query as well.

A logical condition statement is composed of one or more comparison and logical operation statements.

A comparison statement takes the form: `comp(attr, val)`:
- `comp` ({allowed_comparators}): comparator
- `attr` (string):  name of attribute to apply the comparison to
- `val` (string): is the comparison value

A logical operation statement takes the form `op(statement1, statement2, ...)`:
- `op` ({allowed_operators}): logical operator
- `statement1`, `statement2`, ... (comparison statements or logical operation statements): one or more statements to apply the operation to

There is only one "and" logical operation statement and it is connecting all comparison statements (e.g. and(...)).
The list of allowed operators ("{allowed_operators}") is very limited for a reason.
Make sure that you only use the comparators and logical operators listed above ("{allowed_operators}") and no others (not allowed are "or", ...).
If the query cannot be expressed with the allowed operators (only using "{allowed_operators}"), return "NO_FILTER" for the filter value.
If the query requires a forbidden operator like "or", return "NO_FILTER" for the filter value.
Make sure that filters only refer to attributes that exist in the data source.
The query may contain references to attributes outside of the data source given in the last example,
 the user writing the query is very dumb and doesn't know the data source, therefore return "NO_FILTER" for the filter value.
Don't make any assumptions about the data source and the attributes in it, use only the stated information, if there are discrepancies return "NO_FILTER" for the filter value.
Always stick strictly to the value type of each attribute when writing the filter!
Make sure that filters only use the attributed names with its function names if there are functions applied on them.
Make sure that filters only use format `YYYY-MM-DD` when handling date data typed values.
Make sure that filters take into account the descriptions of attributes and only make comparisons that are feasible given the type of data being stored.
Make sure that filters are only used as needed. If there are no filters that should be applied return "NO_FILTER" for the filter value.
\
"""
DEFAULT_SCHEMA_PROMPT = PromptTemplate.from_template(DEFAULT_SCHEMA)

DEFAULT_PREFIX = """\
Your goal is to structure the user's query to match the request schema provided below.

{schema}\
"""

DEFAULT_SUFFIX = """\
<< Example {i}. >>
Data Source:
```json
{{{{
    "content": "{content}",
    "attributes": {attributes}
}}}}
```

User Query:
{{query}}

Structured Request:
"""

NO_FILTER_ANSWER = """\
```json
{{
    "query": "",
    "filter": "NO_FILTER",
    "reason_for_no_filter": "string describing why no filter can be applied given the query and the constraints"
}}
```\
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
llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
output_parser = StructuredQueryOutputParserWithFailReason.from_components(
    allowed_operators=[Operator.AND, Operator.NOT],
    allowed_comparators=allowed_comparators,
    fix_invalid=True,
)

def query_to_structured_filter(unstructured_query: str,
                               document_content_description: str,
                               metadata_field_info: List[AttributeInfo]) -> dict:
    default_schema_prompt = DEFAULT_SCHEMA_PROMPT
    schema_prompt = default_schema_prompt
    attribute_str = _format_attribute_info(metadata_field_info)
    schema = schema_prompt.format(
        allowed_comparators=" | ".join(allowed_comparators),
        allowed_operators=" | ".join([Operator.AND, Operator.NOT]),
    )
    examples = [
        {
            "i": 1,
            "data_source": SONG_DATA_SOURCE,
            "user_query": "What are songs by Taylor Swift about teenage romance under 3 minutes long in the dance pop genre",
            "structured_request": """\
    ```json
    {{
        "query": "teenager love",
        "filter": "and(eq(\\"artist\\", \\"Taylor Swift\\"), lt(\\"length\\", 180), eq(\\"genre\\", \\"pop\\"))"
    }}
    ```\
    """,
        },
        {
            "i": 2,
            "data_source": SONG_DATA_SOURCE,
            "user_query": "What are songs by Taylor Swift or Katy Perry about teenage romance under 3 minutes long in the dance pop genre",
            "structured_request": NO_FILTER_ANSWER,
        },
        {
            "i": 3,
            "data_source": SONG_DATA_SOURCE,
            "user_query": "What are songs that were not published on Spotify",
            "structured_request": NO_FILTER_ANSWER,
        },
    ]
    example_prompt = EXAMPLE_PROMPT
    prefix = DEFAULT_PREFIX.format(schema=schema)
    suffix = DEFAULT_SUFFIX.format(
        i=len(examples) + 1, content=document_content_description, attributes=attribute_str
    )
    prompt = FewShotPromptTemplate(
        examples=list(examples),
        example_prompt=example_prompt,
        input_variables=["query"],
        suffix=suffix,
        prefix=prefix,
    )
    query_constructor = prompt | llm | output_parser
    res = query_constructor.invoke(
        {
            "query": unstructured_query
        }
    )
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


