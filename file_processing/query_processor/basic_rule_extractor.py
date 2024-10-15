# add env vars before loading modules dependent on them (e.g. LLM API)
import logging
from typing import List

"""
Thanks for the idea, info and classes:
https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/self_query/
"""

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

from langchain_core.utils.json import parse_json_markdown

from file_processing.llm_chat_support import get_llm, LLMTemp
from langchain.chains.query_constructor.base import (
    _format_attribute_info,
)
from langchain.chains.query_constructor.schema import AttributeInfo

# Thank you, ChatGPT for adding markdown formatting and helping understand which parts I need to emphasise.
MEGA_PROMPT = """\
### Goal:
Format the user’s query using the schema below.
Note: Use sample values for context only — do not use them directly in the response.


### Request Schema:
Return a markdown code snippet in the following JSON format:

```json
{
    "filter": "list \ list of comparisons applied with logical 'and'",
    "reason_for_no_filter": "string \ explanation if no filter is applied"
}
```

Each comparison in the list of comparisons applied with logical 'and' has the following format:
```json
{
    "value": "string \ the comparison value",
    "attribute": "string \ attribute name from the data source",
    "comparator": "string \ comparator for value and attribute - one of (`eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `contain`, `not_contain`, `like`, `not_like`, `in`, `nin`, `regex_match`, `not_regex_match`)"
}
```

### Key Rules:
- **Query**: Only contains text expected to match document content.
- **Filter**:
  - All comparisons are **always** connected with logical `and`.
  - The only supported comparators are (`eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `contain`, `not_contain`, `like`, `not_like`, `in`, `nin`, `regex_match`, `not_regex_match`).
  - Only (`eq`, `ne`, `gt`, `gte`, `lt`, `lte`) can be applied to number types, other comparators are invalid for number types
  - Only (`eq`, `ne`, `contain`, `not_contain`, `like`, `not_like`, `in`, `nin`, `regex_match`, `not_regex_match`) can be applied to strings, other comparators are invalid for string types
  - (`in`, `nin`) can be applied to all arrays
  - (`contain`, `not_contain`, `like`, `not_like`, `in`, `nin`, `regex_match`, `not_regex_match`) can only be applied to arrays of strings, where the comparator is checked against each string in the array
  - Logical 'OR' can usually be augmented with `regex_match`, `not_regex_match`.
- **No Filter**: Set `reason_for_no_filter` to empty `[]` and set `reason_for_no_filter` to the reason no filter can be applied if (one or many of the conditions below):
  - **Most Important, Check First**: The query tries to use unknown data source attributes (explain).
  - The query is too broad (explain).
  - The query is unrelated to data source attributes (explain).
  - The query can't be represented with the current constraints (explain).

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
    "filter": [
        {
            "value": "Taylor Swift",
            "attribute": "artist",
            "comparator": "eq"
        },
        {
            "value": 180,
            "attribute": "length",
            "comparator": "lt"
        },
        {
            "value": "pop",
            "attribute": "genre",
            "comparator": "eq"
        }
    ]
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
    "filter": [
        {
            "value": "(Taylor Swift|Katy Perry)",
            "attribute": "artist",
            "comparator": "regex_match"
        },
        {
            "value": 180,
            "attribute": "length",
            "comparator": "lt"
        },
        {
            "value": "pop",
            "attribute": "genre",
            "comparator": "eq"
        }
    ]
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
    "filter": [],
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


def query_to_structured_filter(unstructured_query: str,
                               document_content_description: str,
                               metadata_field_info: List[AttributeInfo]) -> dict:
    attribute_str = _format_attribute_info(metadata_field_info)

    suffix = DEFAULT_SUFFIX.format(
        content=document_content_description,
        attributes=attribute_str,
        query=unstructured_query
    ).replace("{{", "{").replace("}}", "}")

    res = get_llm(LLMTemp.ABSTRACT).invoke(MEGA_PROMPT + suffix)
    js = parse_json_markdown(res.content)
    if isinstance(js, dict):
        if len(js.get("filter", [])) > 0:
            return js

    return {
        "reason_for_no_filter": js.get("reason_for_no_filter", "Unknown reason for failed metadata rule extraction")
    }


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
        "What are some movies from the 90's not directed by Luc Besson and the director name should contain Angle or Boris or Travolta + in the sci-fi genre with a rating of at least 5?",
        document_content_description,
        metadata_field_info
    )
    print("res:", res)
