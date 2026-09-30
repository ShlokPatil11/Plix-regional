from dataclasses import dataclass

@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    patch_color: str | None = None

@dataclass
class SubtitleStyle:
    font_name: str = 'Arial'
    font_size: float = 42
    bold: bool = False
    italic: bool = False
    primary_color: str = '#FFFFFF'
    outline_color: str = '#000000'
    outline_width: float = 2
    shadow: float = 1
    alignment: int = 2
    x: float | None = None
    y: float | None = None

@dataclass
class SubtitleSegment:
    id: int
    start: float
    end: float
    source_text: str
    translated_text: str | None = None
    bbox: BoundingBox | None = None
    style: SubtitleStyle | None = None
    ocr_confidence: float | None = None
    event_index: int | None = None
    fit_width: float | None = None
    fit_height: float | None = None
