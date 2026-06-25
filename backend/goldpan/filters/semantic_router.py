import os
from google import genai
from google.genai import types
from goldpan.core.schemas import RawContent, FilteredContent
from dotenv import load_dotenv

load_dotenv()

class SemanticRouterFilter:
    def __init__(self, use_api: bool = True, api_key: str = None):
        self.use_api = use_api
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if self.use_api and self.api_key:
            # Use the new Google GenAI SDK
            self.client = genai.Client(api_key=self.api_key)
            self.model_name = 'gemini-2.5-flash'
        else:
            self.client = None

    def apply(self, content: RawContent | FilteredContent) -> FilteredContent:
        text = content.text
        
        # Fallback về Local Heuristic (bộ lọc chạy bằng rule cơ bản)
        # nếu người dùng không nhập API key hoặc file quá nhỏ (< 1000 ký tự)
        if not self.client or len(text) < 1000:
            return self._apply_local(content)
            
        return self._apply_ai(content)

    def _apply_local(self, content):
        text = content.text
        paragraphs = text.split('\n\n')
        filtered_paragraphs = []
        noise_removed = 0
        
        for p in paragraphs:
            p_strip = p.strip()
            # Local Regex: Bỏ qua các đoạn quá ngắn và không có dấu câu (thường là menu, nút bấm)
            if len(p_strip) < 30 and not any(char in p_strip for char in ['.', '!', '?', ':']):
                noise_removed += len(p)
                continue
            filtered_paragraphs.append(p)
            
        final_text = '\n\n'.join(filtered_paragraphs)
        stats = content.stats.copy() if isinstance(content, FilteredContent) else {}
        stats['semantic_noise_removed_bytes'] = noise_removed
        stats['router_mode'] = 'local_heuristic'
        
        return FilteredContent(text=final_text, doc_info=content.doc_info, stats=stats)

    def _apply_ai(self, content):
        text = content.text
        
        # Dạy con Agent cực kỳ khắt khe (System Prompt)
        system_prompt = """You are an elite Data Cleansing Agent for a RAG pipeline. 
Your ONLY job is to filter out noise, boilerplate, UI menus, advertisements, footers, and useless fragments from the provided document text, keeping ONLY the core, valuable knowledge.
Output the exact same text, but completely remove paragraphs that are just noise.
CRITICAL RULES:
1. DO NOT summarize. 
2. DO NOT rewrite the original text. 
3. DO NOT add conversational text like 'Here is the cleaned text'.
4. Preserve all Markdown formatting (headers, lists, tables).
Just output the clean Markdown text."""

        import concurrent.futures

        # Chia chunk để tránh lỗi vượt quá Output Token Limit của LLM (8192 tokens)
        chunk_size = 15000 # Khoảng ~3000 tokens mỗi block
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        cleaned_chunks = [None] * len(chunks)
        
        def process_chunk(idx, chunk):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=f"{system_prompt}\n\n=== RAW DOCUMENT CHUNK START ===\n{chunk}\n=== RAW DOCUMENT CHUNK END ===",
                    config=types.GenerateContentConfig(
                        temperature=0.0, # Ép AI làm việc như một cái máy rập khuôn, cấm sáng tạo
                    )
                )
                c_text = response.text.strip()
                
                # Cắt bỏ các thẻ bọc Markdown nếu AI lỡ sinh ra
                if c_text.startswith("```markdown"): c_text = c_text[11:]
                elif c_text.startswith("```"): c_text = c_text[3:]
                if c_text.endswith("```"): c_text = c_text[:-3]
                
                return idx, c_text.strip()
            except Exception as e:
                print(f"[SemanticRouter] Gemini API Failed on chunk {idx}: {e}")
                # Fallback: Giữ nguyên chunk đó nếu API lỡ bị lỗi (Rate limit, safety...)
                return idx, chunk
                
        # Run up to 10 threads concurrently to massively speed up extraction
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_chunk, i, chunk) for i, chunk in enumerate(chunks)]
            for future in concurrent.futures.as_completed(futures):
                idx, res = future.result()
                cleaned_chunks[idx] = res
                
        final_text = '\n\n'.join(cleaned_chunks)
        
        stats = content.stats.copy() if isinstance(content, FilteredContent) else {}
        stats['semantic_noise_removed_bytes'] = len(text) - len(final_text)
        stats['router_mode'] = 'gemini_api'
        
        return FilteredContent(text=final_text, doc_info=content.doc_info, stats=stats)
