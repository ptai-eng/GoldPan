import re
from goldpan.core.schemas import RawContent, FilteredContent

class HeuristicFilter:
    def __init__(self):
        # Common boilerplate patterns
        self.noise_patterns = [
            re.compile(r'(?i)cookie policy'),
            re.compile(r'(?i)subscribe to newsletter'),
            re.compile(r'(?i)all rights reserved'),
            re.compile(r'^\s*\[.*\]\(.*\)\s*$') # Loose regex for navigation links standalone
        ]

    def apply(self, content: RawContent | FilteredContent) -> FilteredContent:
        text = content.text
        lines = text.split('\n')
        
        filtered_lines = []
        noise_removed = 0
        
        for line in lines:
            if any(p.search(line) for p in self.noise_patterns):
                noise_removed += len(line)
                continue
            filtered_lines.append(line)
            
        final_text = '\n'.join(filtered_lines)
        
        # Merge stats if it's already a FilteredContent
        stats = content.stats.copy() if isinstance(content, FilteredContent) else {}
        stats['heuristic_noise_removed_bytes'] = noise_removed
        
        return FilteredContent(
            text=final_text,
            doc_info=content.doc_info,
            stats=stats
        )
