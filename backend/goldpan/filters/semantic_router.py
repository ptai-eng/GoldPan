import os
import re
import concurrent.futures
from google import genai
from google.genai import types
from goldpan.core.schemas import RawContent, FilteredContent


def _split_preserving_fences(text: str) -> list[str]:
    """
    Splits Markdown text by double newlines (`\\n\\n`) while keeping fenced code
    blocks (``` or ~~~) intact even when they contain internal blank lines.
    """
    lines = text.split("\n")
    blocks: list[str] = []
    current_lines: list[str] = []
    in_fence = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            current_lines.append(line)
            continue

        if not in_fence and stripped == "":
            if current_lines and current_lines[-1] != "":
                current_lines.append("")
            continue

        if not in_fence and current_lines and current_lines[-1] == "":
            # Flush previous block before starting new block
            block_text = "\n".join(current_lines[:-1]).strip()
            if block_text:
                blocks.append(block_text)
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        block_text = "\n".join(current_lines).strip()
        if block_text:
            blocks.append(block_text)

    return blocks


def _is_structural_markdown(block: str) -> bool:
    """
    Returns True if a short block is a structural Markdown element that must NEVER
    be stripped by paragraph-length heuristics (headings, tables, code fences, lists, blockquotes).
    """
    s = block.lstrip()
    if not s:
        return False
    if s.startswith(("#", "|", "```", "~~~", ">", "- ", "* ", "+ ")):
        return True
    if re.match(r"^\d+\.\s+", s):
        return True
    return False


class SemanticRouterFilter:
    def __init__(self, use_api: bool = True, api_key: str = None):
        self.use_api = use_api
        self.api_key = api_key

        if self.use_api and self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            self.model_name = "gemini-2.5-flash"
        else:
            self.client = None

    def apply(self, content: RawContent | FilteredContent) -> FilteredContent:
        text = content.text

        # Fallback to Local Heuristic when no API key is configured or document is < 1000 chars
        if not self.client or len(text) < 1000:
            return self._apply_local(content)

        return self._apply_ai(content)

    def _apply_local(self, content: RawContent | FilteredContent) -> FilteredContent:
        text = content.text
        blocks = _split_preserving_fences(text)
        filtered_blocks = []
        noise_removed_bytes = 0

        for b in blocks:
            b_strip = b.strip()
            if not b_strip:
                continue
            # Always preserve structural Markdown (headings, tables, code blocks, lists, blockquotes)
            if _is_structural_markdown(b_strip):
                filtered_blocks.append(b)
                continue
            # Drop short non-punctuated UI fragments (buttons,孤立 menu labels)
            if len(b_strip) < 30 and not any(char in b_strip for char in [".", "!", "?", ":", ";"]):
                noise_removed_bytes += len(b.encode("utf-8"))
                continue
            filtered_blocks.append(b)

        final_text = "\n\n".join(filtered_blocks)
        stats = content.stats.copy() if isinstance(content, FilteredContent) else {}
        stats["semantic_noise_removed_bytes"] = noise_removed_bytes
        stats["router_mode"] = "local_heuristic"

        return FilteredContent(text=final_text, doc_info=content.doc_info, stats=stats)

    def _group_blocks_into_chunks(self, text: str, max_chunk_chars: int = 15000) -> list[str]:
        blocks = _split_preserving_fences(text)
        chunks: list[str] = []
        current = ""
        for b in blocks:
            if len(current) + len(b) + 2 <= max_chunk_chars:
                current = f"{current}\n\n{b}" if current else b
            else:
                if current:
                    chunks.append(current)
                if len(b) <= max_chunk_chars:
                    current = b
                else:
                    for i in range(0, len(b), max_chunk_chars):
                        chunks.append(b[i : i + max_chunk_chars])
                    current = ""
        if current:
            chunks.append(current)
        return chunks

    def _apply_ai(self, content: RawContent | FilteredContent) -> FilteredContent:
        text = content.text

        system_prompt = """You are an elite Data Cleansing Agent for a RAG pipeline. 
Your ONLY job is to filter out noise, boilerplate, UI menus, advertisements, footers, and useless fragments from the provided document text, keeping ONLY the core, valuable knowledge.
Output the exact same text, but completely remove paragraphs that are just noise.
CRITICAL RULES:
1. DO NOT summarize. 
2. DO NOT rewrite the original text. 
3. DO NOT add conversational text like 'Here is the cleaned text'.
4. Preserve all Markdown formatting (headers, lists, tables, code blocks).
Just output the clean Markdown text."""

        chunks = self._group_blocks_into_chunks(text, max_chunk_chars=15000)
        cleaned_chunks = [None] * len(chunks)

        def process_chunk(idx: int, chunk: str):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=f"{system_prompt}\n\n=== RAW DOCUMENT CHUNK START ===\n{chunk}\n=== RAW DOCUMENT CHUNK END ===",
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                    ),
                )
                c_text = (response.text or "").strip()

                if c_text.startswith("```markdown"):
                    c_text = c_text[11:]
                elif c_text.startswith("```"):
                    c_text = c_text[3:]
                if c_text.endswith("```"):
                    c_text = c_text[:-3]

                return idx, c_text.strip()
            except Exception as e:
                print(f"[SemanticRouter] Gemini API Failed on chunk {idx}: {e}")
                return idx, chunk

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_chunk, i, chunk) for i, chunk in enumerate(chunks)]
            for future in concurrent.futures.as_completed(futures):
                idx, res = future.result()
                cleaned_chunks[idx] = res

        final_text = "\n\n".join(c for c in cleaned_chunks if c)

        stats = content.stats.copy() if isinstance(content, FilteredContent) else {}
        removed_bytes = len(text.encode("utf-8")) - len(final_text.encode("utf-8"))
        stats["semantic_noise_removed_bytes"] = max(0, removed_bytes)
        stats["router_mode"] = "gemini_api"

        return FilteredContent(text=final_text, doc_info=content.doc_info, stats=stats)
