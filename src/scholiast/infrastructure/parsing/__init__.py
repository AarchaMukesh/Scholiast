"""PDF -> structured document adapters (doc 5 §5.3).

Two tools, one job each (D19/D39): GROBID for structure and coordinates,
PyMuPDF for the cheap local pre-flight that decides a PDF is hopeless before a
GROBID round-trip is spent on it.
"""

from scholiast.infrastructure.parsing.grobid import PARSER_VERSION, GrobidParser
from scholiast.infrastructure.parsing.preflight import MAX_PDF_BYTES, preflight

__all__ = ["MAX_PDF_BYTES", "PARSER_VERSION", "GrobidParser", "preflight"]
