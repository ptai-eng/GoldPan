import gradio as gr
from goldpan.core.pipeline import GoldPanPipeline
from goldpan.extractors.markitdown_wrapper import MarkItDownExtractor
from goldpan.extractors.liteparse_wrapper import LiteParseExtractor
from goldpan.extractors.web_extractor import HybridWebExtractor
from goldpan.filters.heuristic_filter import HeuristicFilter
from goldpan.filters.semantic_router import SemanticRouterFilter
from goldpan.formatters.markdown_formatter import AIFriendlyMarkdownFormatter
import os

def process_input(file_obj, url):
    source = None
    extractor = None
    
    if url and url.strip():
        source = url.strip()
        extractor = HybridWebExtractor(use_playwright_fallback=True)
    elif file_obj:
        source = file_obj.name
        if source.lower().endswith('.pdf'):
            extractor = LiteParseExtractor()
        else:
            extractor = MarkItDownExtractor()
    else:
        return "Please provide either a file or a URL."

    pipeline = GoldPanPipeline(extractor=extractor)
    pipeline.add_filter(HeuristicFilter())
    pipeline.add_filter(SemanticRouterFilter())
    pipeline.set_formatter(AIFriendlyMarkdownFormatter())
    
    try:
        result = pipeline.process(source)
        return result.text
    except Exception as e:
        return f"Error processing: {str(e)}"

with gr.Blocks(title="GoldPan - RAG Data Cleansing") as demo:
    gr.Markdown("# ⛏️ GoldPan (Kẻ Đào Vàng)")
    gr.Markdown("Chuyển đổi dữ liệu thô (Web, PDF, DOCX) thành Markdown siêu sạch cho AI.")
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload File (PDF, DOCX, XLSX, PPTX)")
            url_input = gr.Textbox(label="Or enter URL")
            process_btn = gr.Button("Extract & Clean", variant="primary")
            
        with gr.Column():
            # Use Textbox for raw markdown view, and Markdown for rendered view
            output_md = gr.Textbox(label="Cleaned Markdown (Raw)", lines=20)
            
    process_btn.click(
        fn=process_input,
        inputs=[file_input, url_input],
        outputs=[output_md]
    )

if __name__ == "__main__":
    demo.launch()
