"""Experimental **text splitter** based on semantic similarity."""
import copy
import random
import re
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, cast

import numpy as np
from langchain_community.utils.math import (
    cosine_similarity,
)
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.embeddings import Embeddings


def combine_sentences(sentences: List[dict], buffer_size: int = 1) -> List[dict]:
    """Combine sentences based on buffer size.

    Args:
        sentences: List of sentences to combine.
        buffer_size: Number of sentences to combine. Defaults to 1.

    Returns:
        List of sentences with combined sentences.
    """

    # Go through each sentence dict
    for i in range(len(sentences)):
        # Create a string that will hold the sentences which are joined
        combined_sentence = ""

        # Add sentences before the current one, based on the buffer size.
        for j in range(i - buffer_size, i):
            # Check if the index j is not negative
            # (to avoid index out of range like on the first one)
            if j >= 0:
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += sentences[j]["sentence"] + " "

        # Add the current sentence
        combined_sentence += sentences[i]["sentence"]

        # Add sentences after the current one, based on the buffer size
        for j in range(i + 1, i + 1 + buffer_size):
            # Check if the index j is within the range of the sentences list
            if j < len(sentences):
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += " " + sentences[j]["sentence"]

        # Then add the whole thing to your dict
        # Store the combined sentence in the current sentence dict
        sentences[i]["combined_sentence"] = combined_sentence

    return sentences


def calculate_cosine_distances(sentences: List[dict]) -> Tuple[List[float], List[dict]]:
    """Calculate cosine distances between sentences.

    Args:
        sentences: List of sentences to calculate distances for.

    Returns:
        Tuple of distances and sentences.
    """
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]["combined_sentence_embedding"]
        embedding_next = sentences[i + 1]["combined_sentence_embedding"]

        # Calculate cosine similarity
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]

        # Convert to cosine distance
        distance = 1 - similarity

        # Append cosine distance to the list
        distances.append(distance)

        # Store distance in the dictionary
        sentences[i]["distance_to_next"] = distance

    # Optionally handle the last sentence
    # sentences[-1]['distance_to_next'] = None  # or a default value

    return distances, sentences


BreakpointThresholdType = Literal["percentile", "standard_deviation", "interquartile"]
BREAKPOINT_DEFAULTS: Dict[BreakpointThresholdType, float] = {
    "percentile": 95,
    "standard_deviation": 3,
    "interquartile": 1.5,
}


class SemanticChunker(BaseDocumentTransformer):
    """Split the text based on semantic similarity.

    Taken from Greg Kamradt's wonderful notebook:
    https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb

    All credits to him.

    At a high level, this splits into sentences, then groups into groups of 3
    sentences, and then merges one that are similar in the embedding space.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        buffer_size: int = 1,
        add_start_index: bool = False,
        breakpoint_threshold_type: BreakpointThresholdType = "percentile",
        breakpoint_threshold_amount: Optional[float] = None,
        number_of_chunks: Optional[int] = None,
        sentence_split_regex: str = r"(?<=[.?!])\s+",
    ):
        self._add_start_index = add_start_index
        self.embeddings = embeddings
        self.buffer_size = buffer_size
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.number_of_chunks = number_of_chunks
        self.sentence_split_regex = sentence_split_regex
        if breakpoint_threshold_amount is None:
            self.breakpoint_threshold_amount = BREAKPOINT_DEFAULTS[
                breakpoint_threshold_type
            ]
        else:
            self.breakpoint_threshold_amount = breakpoint_threshold_amount

    def _calculate_breakpoint_threshold(self, distances: List[float]) -> float:
        if self.breakpoint_threshold_type == "percentile":
            return cast(
                float,
                np.percentile(distances, self.breakpoint_threshold_amount),
            )
        elif self.breakpoint_threshold_type == "standard_deviation":
            return cast(
                float,
                np.mean(distances)
                + self.breakpoint_threshold_amount * np.std(distances),
            )
        elif self.breakpoint_threshold_type == "interquartile":
            q1, q3 = np.percentile(distances, [25, 75])
            iqr = q3 - q1

            return np.mean(distances) + self.breakpoint_threshold_amount * iqr
        else:
            raise ValueError(
                f"Got unexpected `breakpoint_threshold_type`: "
                f"{self.breakpoint_threshold_type}"
            )

    def _threshold_from_clusters(self, distances: List[float]) -> float:
        """
        Calculate the threshold based on the number of chunks.
        Inverse of percentile method.
        """
        if self.number_of_chunks is None:
            raise ValueError(
                "This should never be called if `number_of_chunks` is None."
            )
        x1, y1 = len(distances), 0.0
        x2, y2 = 1.0, 100.0

        x = max(min(self.number_of_chunks, x1), x2)

        # Linear interpolation formula
        y = y1 + ((y2 - y1) / (x2 - x1)) * (x - x1)
        y = min(max(y, 0), 100)

        return cast(float, np.percentile(distances, y))

    def _calculate_sentence_distances(
        self, single_sentences_list: List[str]
    ) -> Tuple[List[float], List[dict]]:
        """Split text into multiple components."""

        _sentences = [
            {"sentence": x, "index": i} for i, x in enumerate(single_sentences_list)
        ]
        sentences = combine_sentences(_sentences, self.buffer_size)
        embeddings = self.embeddings.embed_documents(
            [x["combined_sentence"] for x in sentences]
        )
        for i, sentence in enumerate(sentences):
            sentence["combined_sentence_embedding"] = embeddings[i]

        return calculate_cosine_distances(sentences)

    def split_text_deprecated(
        self,
        text: str,
    ) -> List[str]:
        # Splitting the essay (by default on '.', '?', and '!')
        single_sentences_list = re.split(self.sentence_split_regex, text)

        # having len(single_sentences_list) == 1 would cause the following
        # np.percentile to fail.
        if len(single_sentences_list) == 1:
            return single_sentences_list
        distances, sentences = self._calculate_sentence_distances(single_sentences_list)
        if self.number_of_chunks is not None:
            breakpoint_distance_threshold = self._threshold_from_clusters(distances)
        else:
            breakpoint_distance_threshold = self._calculate_breakpoint_threshold(
                distances
            )

        indices_above_thresh = [
            i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
        ]

        chunks = []
        start_index = 0

        # Iterate through the breakpoints to slice the sentences
        for index in indices_above_thresh:
            # The end index is the current breakpoint
            end_index = index

            # Slice the sentence_dicts from the current start index to the end index
            group = sentences[start_index : end_index + 1]
            combined_text = " ".join([d["sentence"] for d in group])
            chunks.append(combined_text)

            # Update the start index for the next group
            start_index = index + 1

        # The last group, if any sentences remain
        if start_index < len(sentences):
            combined_text = " ".join([d["sentence"] for d in sentences[start_index:]])
            chunks.append(combined_text)
        return chunks

    def _find_optimal_split(self, sentences, start_index, end_index, max_length):
        """
        Find the optimal split by examining various semantic distance thresholds at weighted random percentiles.
        The split aims to approximate a maximum chapter length in characters.
        """
        # Calculate cumulative character lengths
        cumulative_lengths = np.cumsum([len(sentences[i]["sentence"]) for i in range(start_index, end_index)])

        # Collect semantic distances for this segment
        segment_distances = [sentences[i]["distance_to_next"] for i in range(start_index, end_index - 1)]

        # Define percentiles and corresponding weights
        percentiles = list(range(95, 10, -5))
        """ 
        # Weighted Random Sampling potentially, if we want to try lower percentiles faster
        weights = [i ** 2 for i in range(len(percentiles), 0, -1)]  # Weight higher percentiles more heavily

        # Sample percentiles based on weights
        sampled_percentiles = random.choices(percentiles, weights, k=10)
        for perc in sorted(sampled_percentiles, reverse=True):
        """
        for perc in percentiles:
            threshold = np.percentile(segment_distances, perc)
            potential_splits = [i for i, dist in enumerate(segment_distances) if dist > threshold]

            # Find the index where the split best balances the chapter size
            if potential_splits:
                # Convert index to the actual index in the sentences list and adjust by character length
                closest_split_index = min(potential_splits, key=lambda x: abs(cumulative_lengths[x] - max_length // 2))
                return start_index + closest_split_index

        # If no semantic split point is found, fall back to a middle split
        return start_index + len(segment_distances) // 2

    def _recursive_split(self, sentences, start_index, end_index, max_length, cumulative_lengths):
        """Recursively split large chapters until all parts are under max_length."""
        chunks = []
        if (cumulative_lengths[end_index - 1] - (
        cumulative_lengths[start_index - 1] if start_index > 0 else 0)) <= max_length:
            # If within max_length, no further split needed, just return the combined text
            combined_text = " ".join(d["sentence"] for d in sentences[start_index:end_index])
            chunks.append(combined_text)
        else:
            # Find an optimal split point
            optimal_split = self._find_optimal_split(sentences, start_index, end_index, max_length)
            # Split the first half recursively
            chunks += self._recursive_split(sentences, start_index, optimal_split + 1, max_length, cumulative_lengths)
            # Split the second half recursively
            chunks += self._recursive_split(sentences, optimal_split + 1, end_index, max_length, cumulative_lengths)
        return chunks

    def split_text(self, text: str, min_length: int = 300, max_length: int = 1500) -> List[str]:
        single_sentences_list = re.split(self.sentence_split_regex, text)

        if len(single_sentences_list) == 1:
            return single_sentences_list

        distances, sentences = self._calculate_sentence_distances(single_sentences_list)
        cumulative_lengths = np.cumsum([len(sentence["sentence"]) for sentence in sentences])

        if self.number_of_chunks is not None:
            breakpoint_distance_threshold = self._threshold_from_clusters(distances)
        else:
            breakpoint_distance_threshold = self._calculate_breakpoint_threshold(distances)

        indices_above_thresh = [
                                   i for i, dist in enumerate(distances) if dist > breakpoint_distance_threshold
                               ] + [len(sentences) - 1]  # Ensure the last segment is included

        chunks = []
        start_index = 0

        for index in indices_above_thresh:
            chunks += self._recursive_split(sentences, start_index, index + 1, max_length, cumulative_lengths)
            start_index = index + 1

        # Merging small chunks measured by character count
        final_chunks = []
        current_chunk = chunks[0]

        for chunk in chunks[1:]:
            if len(current_chunk) < min_length:
                current_chunk += " " + chunk
            else:
                final_chunks.append(current_chunk)
                current_chunk = chunk

        if current_chunk:
            final_chunks.append(current_chunk)

        return final_chunks

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            index = -1
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])
                if self._add_start_index:
                    index = text.find(chunk, index + 1)
                    metadata["start_index"] = index
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """Split documents."""
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Transform sequence of documents by splitting them."""
        return self.split_documents(list(documents))
