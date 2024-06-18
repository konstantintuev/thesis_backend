def concat_chunks(chunks: list[str], min_length: int, max_length: int or None) -> list[str]:
    def is_complete_sentence(text: str) -> bool:
        return text.rstrip().endswith(('.', '!', '?'))

    concatenated_chunks = []
    i = 0

    while i < len(chunks):
        current_chunk = chunks[i].strip()

        # Check if current chunk is too short or doesn't end with a complete sentence
        while ((len(current_chunk) < min_length or not is_complete_sentence(current_chunk))
               and i < len(chunks) - 1):
            next_chunk = chunks[i + 1].strip()
            # If the concatted chunks is longer than max_length -> skip
            if max_length is not None and len(current_chunk) + len(next_chunk) + 1 > max_length:
                break
            i += 1
            current_chunk += ' ' + next_chunk

        concatenated_chunks.append(current_chunk)
        i += 1

    return concatenated_chunks