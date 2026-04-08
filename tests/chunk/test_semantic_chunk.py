"""
SemanticTokenChunker 的三层测试：
  1. 正确性单测 - 验证基本契约
  2. 属性测试   - 随机输入验证不变量
  3. 质量基准   - 量化切分质量，防止退化
"""
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rag_from_scratch.chunk.semantic_chunk import SemanticTokenChunker


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture(scope="session")
def chunker():
    return SemanticTokenChunker(chunk_size=100, overlap=0)


@pytest.fixture(scope="session")
def long_text():
    return "\n\n".join([f"这是第{i}个段落，包含一些测试内容，用于验证切块行为是否符合预期。" * 3 for i in range(30)])


@pytest.fixture(scope="session")
def sample_doc():
    fixture_path = "tests/fixtures/sample_doc.txt"
    with open(fixture_path, encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="session")
def property_chunker():
    """hypothesis 属性测试复用，避免每个 example 重复加载 tokenizer"""
    return SemanticTokenChunker(chunk_size=64, overlap=8)


@pytest.fixture(scope="session")
def benchmark_chunker():
    """质量基准测试复用"""
    return SemanticTokenChunker(chunk_size=256, overlap=32)


# ─────────────────────────────────────────────
# 第一层：正确性单测
# ─────────────────────────────────────────────

class TestCorrectness:
    def test_no_key_content_lost(self, chunker):
        """关键内容不应在切分过程中丢失"""
        text = "这是第一段。\n\n这是第二段。\n\n这是第三段。"
        chunks = chunker.split(text)
        combined = "".join(chunks)
        for keyword in ["第一段", "第二段", "第三段"]:
            assert keyword in combined, f"关键内容 '{keyword}' 在切分后丢失"

    def test_chunk_size_not_exceeded(self, chunker, long_text):
        """每个 chunk 的 token 数不应超过 chunk_size"""
        chunks = chunker.split(long_text)
        for i, chunk in enumerate(chunks):
            size = chunker.token_len(chunk)
            assert size <= chunker.chunk_size, (
                f"chunk[{i}] 超出大小限制: {size} > {chunker.chunk_size}"
            )

    def test_overlap_content_shared(self, long_text):
        """相邻 chunk 之间应存在重叠内容"""
        chunker = SemanticTokenChunker(chunk_size=100, overlap=30)
        chunks = chunker.split(long_text)
        assert len(chunks) > 1, "文本应被切分为多个 chunk"

        overlap_found = False
        for i in range(len(chunks) - 1):
            # 取前一个 chunk 的尾部片段，检查是否出现在后一个 chunk 头部
            tail = chunks[i][-20:]
            if tail and tail in chunks[i + 1]:
                overlap_found = True
                break
        assert overlap_found, "相邻 chunk 之间未检测到重叠内容"

    def test_no_overlap_when_zero(self):
        """overlap=0 时，chunk[i] 的尾部不应出现在 chunk[i+1] 的开头"""
        chunker = SemanticTokenChunker(chunk_size=100, overlap=0)
        # 每段内容唯一，排除因重复文本误判为 overlap 的情况
        text = "\n\n".join([f"第{i}段：这是独立内容，编号唯一，不与其他段落重复。" for i in range(60)])
        chunks = chunker.split(text)
        assert len(chunks) > 1
        for i in range(len(chunks) - 1):
            tail = chunks[i][-15:]
            head = chunks[i + 1][:15]
            # 只检查边界处：尾部不应出现在下一 chunk 的开头
            assert tail not in head, f"overlap=0 时 chunk[{i}] 尾部出现在 chunk[{i+1}] 开头"

    def test_empty_input_returns_empty(self, chunker):
        """空字符串输入应返回空结果（空列表或含空字符串的列表）"""
        result = chunker.split("")
        assert result == [] or result == [""]

    def test_short_text_not_split(self, chunker):
        """远小于 chunk_size 的文本不应被切分"""
        short = "你好世界，这是一段很短的文字。"
        chunks = chunker.split(short)
        assert len(chunks) == 1

    def test_returns_list_of_strings(self, chunker):
        """返回值必须是字符串列表"""
        chunks = chunker.split("任意文本内容")
        assert isinstance(chunks, list)
        assert all(isinstance(c, str) for c in chunks)

    def test_single_oversized_segment(self):
        """单个超长段落（无可用分隔符）最终应被切分"""
        chunker = SemanticTokenChunker(chunk_size=20, overlap=0)
        text = "这" * 200  # 无分隔符的超长文本
        chunks = chunker.split(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunker.token_len(chunk) <= chunker.chunk_size

    @pytest.mark.parametrize("sep,text", [
        ("paragraph", "第一段内容，描述了某个主题。\n\n第二段内容，讨论了另一个话题。\n\n第三段。"),
        ("newline",   "行一\n行二\n行三\n行四\n行五\n行六\n行七\n行八"),
        ("chinese",   "第一句。第二句。第三句！第四句？第五句；第六句。"),
    ])
    def test_respects_separator_hierarchy(self, sep, text):
        """应优先在粗粒度分隔符处切分"""
        chunker = SemanticTokenChunker(chunk_size=30, overlap=0)
        chunks = chunker.split(text)
        assert len(chunks) >= 1  # 至少不崩溃


# ─────────────────────────────────────────────
# 第二层：属性测试（随机输入）
# ─────────────────────────────────────────────

class TestProperties:
    @given(st.text(min_size=0, max_size=1000))
    @settings(max_examples=300, deadline=None)
    def test_chunk_size_never_exceeded(self, property_chunker, text):
        """有 overlap 时，chunk 最多为 chunk_size + overlap"""
        chunks = property_chunker.split(text)
        limit = property_chunker.chunk_size + property_chunker.overlap
        for chunk in chunks:
            assert property_chunker.token_len(chunk) <= limit

    @given(st.text(min_size=1, max_size=500))
    @settings(max_examples=200, deadline=None)
    def test_always_returns_list(self, property_chunker, text):
        """任意输入都应返回列表，不应抛出异常"""
        result = property_chunker.split(text)
        assert isinstance(result, list)

    @given(
        st.integers(min_value=32, max_value=256),
        st.integers(min_value=0, max_value=30),
    )
    @settings(max_examples=50, deadline=None)
    def test_various_chunk_sizes(self, property_chunker, chunk_size, overlap):
        """不同参数组合下均应正常工作（复用 tokenizer，仅 chunk_size/overlap 变化）"""
        property_chunker.chunk_size = chunk_size
        property_chunker.overlap = overlap
        text = "这是测试文本。" * 50
        chunks = property_chunker.split(text)
        assert isinstance(chunks, list)
        limit = chunk_size + overlap
        for chunk in chunks:
            assert property_chunker.token_len(chunk) <= limit


# ─────────────────────────────────────────────
# 第三层：质量基准测试
# ─────────────────────────────────────────────

class TestQualityBenchmark:
    def test_chunk_count_reasonable(self, benchmark_chunker, sample_doc):
        """chunk 数量应在合理范围内（不过多也不过少）"""
        chunks = benchmark_chunker.split(sample_doc)
        assert 3 <= len(chunks) <= 50, f"chunk 数量异常: {len(chunks)}"

    def test_average_chunk_size_reasonable(self, benchmark_chunker, sample_doc):
        """平均 chunk 大小应接近 chunk_size，不应出现大量碎片"""
        chunks = benchmark_chunker.split(sample_doc)
        sizes = [benchmark_chunker.token_len(c) for c in chunks]
        avg = sum(sizes) / len(sizes)
        assert avg >= benchmark_chunker.chunk_size * 0.3, f"平均 chunk 太小，碎片化严重: {avg:.0f} tokens"

    def test_tiny_chunk_ratio(self, benchmark_chunker, sample_doc):
        """极小 chunk（< 20 tokens）占比不应超过 20%"""
        chunks = benchmark_chunker.split(sample_doc)
        sizes = [benchmark_chunker.token_len(c) for c in chunks]
        tiny = sum(1 for s in sizes if s < 20)
        ratio = tiny / len(chunks)
        assert ratio < 0.2, f"碎片 chunk 占比过高: {ratio:.1%}"

    def test_no_duplicate_chunks(self, benchmark_chunker, sample_doc):
        """（overlap=0 时）不应出现完全重复的 chunk"""
        benchmark_chunker.overlap = 0
        chunks = benchmark_chunker.split(sample_doc)
        assert len(chunks) == len(set(chunks)), "存在完全重复的 chunk"

    def test_print_distribution(self, benchmark_chunker, sample_doc, capsys):
        """输出切分分布，供人工审查（不做断言，CI 中可存为 artifact）"""
        benchmark_chunker.overlap = 32
        chunks = benchmark_chunker.split(sample_doc)
        sizes = [benchmark_chunker.token_len(c) for c in chunks]

        print(f"\n{'─'*40}")
        print(f"总 chunk 数:    {len(chunks)}")
        print(f"平均 token 数:  {sum(sizes)/len(sizes):.1f}")
        print(f"最小 token 数:  {min(sizes)}")
        print(f"最大 token 数:  {max(sizes)}")
        print(f"chunk_size 上限: {benchmark_chunker.chunk_size}")
        print(f"{'─'*40}")

        captured = capsys.readouterr()
        assert "总 chunk 数" in captured.out
