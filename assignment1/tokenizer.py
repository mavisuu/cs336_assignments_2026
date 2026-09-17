from collections.abc import Iterable, Iterator
import json
from typing import BinaryIO
import regex as re
from collections import Counter, defaultdict
import os

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
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

def pretokenize(
        text: str,
        PAT: str,
        special_tokens: list[str] | None = None,
)-> list[str]:
    pretokens:list[str] = []
    # 先把 special token 和 normal text 分开
    if special_tokens is not None:
        sorted_special_tokens = sorted(
            special_tokens,
            key=len,
            reverse=True,
        )
        special_pattern = "(" + "|".join(re.escape(tok) for tok in sorted_special_tokens) + ")"
        segments = re.split(special_pattern, text)
    else:
        segments = [text]

    # 再处理 normal text 
    for segment in segments:
        if not segment:
            continue
        if special_tokens is not None and segment in special_tokens:
            pretokens.append(segment)
        else:
            for match in re.finditer(PAT, segment):
                pretokens.append(match.group())

    return pretokens
    

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        # special tokens 加入 vocab
        if special_tokens is not None:
            for token in special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes in self.vocab.values():
                    continue
                else:
                    self.vocab[len(self.vocab)] = token_bytes

        # 处理 merges 优先级，避免后续多次遍历，将 merges 镜像为 dict，dict 便于查询
        self.merges_dict:dict[tuple[bytes,bytes],int] = {
            pair: i
            for i, pair in enumerate(self.merges)
        }

        # 将 vocab 镜像，遍于查询 token id
        self.vocab_dict:dict[bytes, int] = {
            b: i
            for i, b in self.vocab.items()
        }


    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
            vocab = {
                int(token_id): bytes.fromhex(token_hex)
                for token_id, token_hex in vocab_data.items()
            }
        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges_data = json.load(f)
            merges = [
                (bytes.fromhex(token1), bytes.fromhex(token2))
                for token1, token2 in merges_data
            ]

        return cls(vocab, merges, special_tokens)

    

    def encode(
        self,
        text: str,
    ) -> list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        res:list[int] = []

        # pre-tokenize
        pretokens = pretokenize(text, PAT, self.special_tokens)

        # 遍历 pretokens，对每个 pretoken 查找 merges 中 rank 最高的 merge 进行合并，直到这个 pretoken 无法被合并
        for pretoken in pretokens:
            if self.special_tokens is not None and pretoken in self.special_tokens:
                res.append(self.vocab_dict[pretoken.encode("utf-8")])
                continue
            pretoken_bytes:list[bytes] = [bytes([b]) for b in pretoken.encode("utf-8")]

            while 1:
                best_rank = len(self.merges) + 8
                best_id = 0
                for i in range(len(pretoken_bytes)-1):
                    pair = (pretoken_bytes[i],pretoken_bytes[i+1]) 
                    if pair in self.merges_dict and self.merges_dict[pair] < best_rank:
                        best_rank = self.merges_dict[pair]
                        best_id = i
                if best_rank == len(self.merges) + 8:
                    break
                else:
                    pretoken_bytes = pretoken_bytes[:best_id] + [pretoken_bytes[best_id]+pretoken_bytes[best_id+1]] + pretoken_bytes[best_id+2:]    

            for b in pretoken_bytes:
                res.append(self.vocab_dict[b])

        return res

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        buffer = ""
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        for text in iterable:
            buffer += text

            # 先 pretokenize 得到完整文本
            text_chunks = pretokenize(buffer, PAT, self.special_tokens)

            # 最后一个 text_chunk 保留作为新 buffer，前面的 chunk 进行 encode
            for i in range(len(text_chunks)-1):
                for id in self.encode(text_chunks[i]):
                    yield id
            if len(text_chunks) > 0:
                buffer = text_chunks[-1]

        for id in self.encode(buffer):
            yield id

    def decode(
        self,
        ids: list[int],
    ) -> str:
        byte_data = b"".join(self.vocab[token_id] for token_id in ids)

        text = byte_data.decode("utf-8", errors="replace")

        return text