from llama_parse import LlamaParse

llama_parser = LlamaParse(
    # can also be set in your env as LLAMA_CLOUD_API_KEY
    result_type="markdown",  # "markdown" and "text" are available
    num_workers=6,  # if multiple files passed, split in `num_workers` API calls
    verbose=True,
    language="en",  # Optionally you can define a language, default=en
)


def pdf_to_md_llama_parse(pdf_filepath: str) -> str | None:
    documents = llama_parser.load_data(pdf_filepath)
    if documents is None or len(documents) == 0:
        return None
    document = documents[0]
    return document.text
