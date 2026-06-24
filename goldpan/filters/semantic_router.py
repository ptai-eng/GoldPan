import os
from goldpan.core.schemas import RawContent, FilteredContent

class SemanticRouterFilter:
    def __init__(self, use_api: bool = False):
        self.use_api = use_api
        # Mock: In a real implementation, this would use local BGE-m3 embeddings 
        # or OpenAI/Gemini API to classify if a chunk is "core knowledge" or "noise"
        
    def apply(self, content: RawContent | FilteredContent) -> FilteredContent:
        text = content.text
        paragraphs = text.split('\n\n')
        
        filtered_paragraphs = []
        noise_removed = 0
        
        for p in paragraphs:
            p_strip = p.strip()
            # Mock Semantic Router: Drop very short paragraphs lacking punctuation
            if len(p_strip) < 30 and not any(char in p_strip for char in ['.', '!', '?', ':']):
                noise_removed += len(p)
                continue
            filtered_paragraphs.append(p)
            
        final_text = '\n\n'.join(filtered_paragraphs)
        
        stats = content.stats.copy() if isinstance(content, FilteredContent) else {}
        stats['semantic_noise_removed_bytes'] = noise_removed
        
        return FilteredContent(
            text=final_text,
            doc_info=content.doc_info,
            stats=stats
        )
