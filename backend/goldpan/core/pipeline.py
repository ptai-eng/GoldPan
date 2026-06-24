from typing import List, Optional, Protocol, Any
from .schemas import RawContent, FilteredContent, FinalMarkdown
from .exceptions import ExtractionError, FilterError, FormattingError

class Extractor(Protocol):
    def extract(self, source: str) -> RawContent:
        ...

class Filter(Protocol):
    def apply(self, content: RawContent | FilteredContent) -> FilteredContent:
        ...

class Formatter(Protocol):
    def format(self, content: FilteredContent) -> FinalMarkdown:
        ...

class GoldPanPipeline:
    def __init__(self, extractor: Extractor):
        self.extractor = extractor
        self.filters: List[Filter] = []
        self.formatter: Optional[Formatter] = None

    def add_filter(self, filter_layer: Filter):
        self.filters.append(filter_layer)

    def set_formatter(self, formatter: Formatter):
        self.formatter = formatter

    def process(self, source: str) -> FinalMarkdown:
        # Step 1: Extraction
        try:
            current_content = self.extractor.extract(source)
        except Exception as e:
            raise ExtractionError(f"Failed to extract from {source}: {str(e)}")

        # Step 2: Filters
        for i, filter_layer in enumerate(self.filters):
            try:
                current_content = filter_layer.apply(current_content)
            except Exception as e:
                raise FilterError(f"Failed at filter layer {i} ({filter_layer.__class__.__name__}): {str(e)}")

        # Step 3: Format
        if not self.formatter:
            raise FormattingError("No formatter set for pipeline.")
            
        try:
            return self.formatter.format(current_content)
        except Exception as e:
            raise FormattingError(f"Failed to format output: {str(e)}")
