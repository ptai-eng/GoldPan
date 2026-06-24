class HybridVisionCaptioner:
    def __init__(self):
        self.local_mode = True

    def suggest_api_if_needed(self, image_count: int, table_count: int) -> str:
        if image_count > 10 or table_count > 5:
            return "WARNING: Large number of complex images/tables detected. Suggest switching to API Vision Model for higher accuracy."
        return "Local Vision Model is sufficient."

    def process_image(self, image_path: str) -> str:
        if self.local_mode:
            return "[Image: Local Vision Model Alt-text generated]"
        else:
            return "[Image: API Vision Model Alt-text generated]"
