from dataclasses import dataclass

ACTIVE = ('ANALYZING','EXTRACTING_SUBTITLES','OCR','TRANSLATING','STYLE_ANALYSIS','RENDERING','QC','PROCESSING')
STATUSES = ('DETECTED','WAITING_FOR_FILE','QUEUED',*ACTIVE,'COMPLETED','FAILED')

@dataclass
class Job:
    job_id: str
    filename: str
    original_path: str
    processing_path: str
    output_path: str | None
    file_hash: str
    source_language: str
    target_language: str
    status: str
    current_stage: str
    retry_count: int
    error_message: str | None
