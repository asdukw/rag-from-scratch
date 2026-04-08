from transformers import AutoTokenizer


class SemanticTokenChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 0):
        self.embed_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = ["\n\n", "\n", "。", "！", "？", "；", "，", " "]

    def token_len(self, text: str) -> int:
        return len(self.embed_tokenizer.encode(text, add_special_tokens=False))

    def split(self, text: str) -> list[str]:
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        sep = None
        for s in separators:
            if s in text:
                sep = s
                break

        if sep is None:
            return self._split_by_tokens(text)

        splits = text.split(sep)

        chunks = []
        current = ""

        for s in splits:
            candidate = current + (sep + s if current and sep else s)

            # 情况1：可以放
            if self.token_len(candidate) <= self.chunk_size:
                current = candidate
                continue

            # 放不下
            if current:
                chunks.append(current)  # 先添加已缓存的

            # 单段过大 → 递归细切
            if self.token_len(s) > self.chunk_size:
                sub_chunks = self._recursive_split(s, separators[1:])
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = s

        if current:
            chunks.append(current)

        # overlap 处理
        if self.overlap > 0:
            chunks = self._apply_overlap(chunks)

        return chunks

    def _split_by_tokens(self, text: str) -> list[str]:
        tokens = self.embed_tokenizer.encode(text, add_special_tokens=False)
        chunks = []

        for i in range(0, len(tokens), self.chunk_size):
            sub_tokens = tokens[i : i + self.chunk_size]
            chunk = self.embed_tokenizer.decode(sub_tokens)
            chunks.append(chunk)

        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        new_chunks = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                new_chunks.append(chunk)
                continue

            prev = new_chunks[-1]

            prev_tokens = self.embed_tokenizer.encode(prev, add_special_tokens=False)
            overlap_tokens = prev_tokens[-self.overlap :] if self.overlap > 0 else []

            current_tokens = self.embed_tokenizer.encode(
                chunk, add_special_tokens=False
            )

            merged = overlap_tokens + current_tokens
            new_chunk = self.embed_tokenizer.decode(merged)

            new_chunks.append(new_chunk)

        return new_chunks
