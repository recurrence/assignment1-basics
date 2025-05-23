import os
from typing import BinaryIO
import regex as re
import multiprocessing as mp
from collections import Counter
from functools import reduce


class PreTokenizer:
    num_processes = 10
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def frequencies_from_file(
        self, input_path: str | os.PathLike, special_tokens: list[str]
    ) -> dict[tuple[bytes, ...], int]:
        with open(input_path, "rb") as f:
            # FYI: In the future chunk across any special tokens
            boundaries = PreTokenizer.__find_chunk_boundaries(
                f, PreTokenizer.num_processes, special_tokens[0].encode("utf-8")
            )  # noqa: UP012

            pool = mp.Pool(processes=self.num_processes)

            chunks: list[str] = []

            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")

                chunks.append(chunk)

            results = pool.starmap(self.frequencies_from_text, ((chunk, special_tokens) for chunk in chunks))

            merged_results = reduce(lambda x, y: x + y, results)

            return merged_results

    def frequencies_from_text(self, text: str, special_tokens: list[str] | None) -> Counter[tuple[bytes, ...]]:
        if special_tokens is not None:
            escaped_special_characters = list(map(re.escape, special_tokens))

            escaped_special_characters_regex = "|".join(escaped_special_characters)

            cleaned_documents = re.split(escaped_special_characters_regex, text)
        else:
            cleaned_documents = [text]

        frequencies = Counter[tuple[bytes, ...]]()

        for document in cleaned_documents:
            tokens = re.finditer(PreTokenizer.PAT, document)
            for token in tokens:
                frequencies[tuple(bytes([byte]) for byte in token.group().encode("utf8"))] += 1

        return frequencies

    def pretokens_from_text(self, text: str, special_tokens: list[str] | None) -> list[tuple[bytes, ...]]:
        if special_tokens is not None:
            # Sort by length in reverse to handle overlapping special tokens
            special_tokens.sort(key=lambda s: len(s), reverse=True)

            escaped_special_characters = list(map(re.escape, special_tokens))
            escaped_special_characters_included_regex = f"({'|'.join(escaped_special_characters)})"

            documents = re.split(escaped_special_characters_included_regex, text)
        else:
            documents = [text]

        response: list[tuple[bytes, ...]] = []

        for document in documents:
            if special_tokens and document in special_tokens:
                response.append(tuple(bytes([byte]) for byte in document.encode("utf8")))
            else:
                tokens = re.finditer(PreTokenizer.PAT, document)
                for token in tokens:
                    response.append(tuple(bytes([byte]) for byte in token.group().encode("utf8")))

        return response

    @classmethod
    def __find_chunk_boundaries(cls, file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))


if __name__ == "__main__":
    print("PreTokenizer: Starting")
    pre_tokenizer = PreTokenizer()
    pre_tokenizer.frequencies_from_file("data/TinyStoriesV2-GPT4-train.txt", ["<|endoftext|>"])
    print("PreTokenizer: Finished")
