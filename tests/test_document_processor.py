import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from document_processor import GeminiDocumentProcessor, ChunkTimeoutError

class DummyModel:
    def generate_content(self, prompt, generation_config=None):
        class Response:
            text = "Summary"
        return Response()

def test_initialize_api(monkeypatch):
    monkeypatch.setattr('google.generativeai.GenerativeModel', lambda name: DummyModel())
    monkeypatch.setattr('google.generativeai.configure', lambda api_key: None)
    processor = GeminiDocumentProcessor(api_key='dummy', model_name='gemini-2.0-flash')
    assert processor.model_name == 'gemini-2.0-flash'

def test_get_total_pages(tmp_path, monkeypatch):
    # Create a dummy PDF file
    pdf_path = tmp_path / "test.pdf"
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF")
    monkeypatch.setattr('pypdf.PdfReader', lambda f: type('R', (), {'pages': [1, 2, 3]})())
    monkeypatch.setattr('google.generativeai.GenerativeModel', lambda name: DummyModel())
    monkeypatch.setattr('google.generativeai.configure', lambda api_key: None)
    processor = GeminiDocumentProcessor(api_key='dummy')
    assert processor.get_total_pages(str(pdf_path)) == 3

def test_get_total_pages_nonexistent():
    processor = GeminiDocumentProcessor(api_key='dummy')
    assert processor.get_total_pages('nonexistent.pdf') == 0

def test_extract_metadata_pdf(monkeypatch, tmp_path):
    pdf_path = tmp_path / "test.pdf"
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF")
    monkeypatch.setattr('pypdf.PdfReader', lambda f: type('R', (), {'metadata': {'/Title': 'Test'}})())
    monkeypatch.setattr('google.generativeai.GenerativeModel', lambda name: DummyModel())
    monkeypatch.setattr('google.generativeai.configure', lambda api_key: None)
    processor = GeminiDocumentProcessor(api_key='dummy')
    meta = processor._extract_pdf_metadata(str(pdf_path))
    assert meta['title'] == 'Test'

def test_extract_text_from_pdf_pages(monkeypatch, tmp_path):
    pdf_path = tmp_path / "test.pdf"
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF")
    class DummyPage:
        def extract_text(self):
            return "Hello"
    class DummyReader:
        pages = [DummyPage(), DummyPage()]
    monkeypatch.setattr('pypdf.PdfReader', lambda f: DummyReader())
    monkeypatch.setattr('google.generativeai.GenerativeModel', lambda name: DummyModel())
    monkeypatch.setattr('google.generativeai.configure', lambda api_key: None)
    processor = GeminiDocumentProcessor(api_key='dummy')
    text = processor._extract_text_from_pdf_pages(str(pdf_path), 1, 2)
    assert "Hello" in text

def test_extract_text_from_pdf_pages_corrupted(monkeypatch, tmp_path):
    pdf_path = tmp_path / "corrupt.pdf"
    with open(pdf_path, "wb") as f:
        f.write(b"not a real pdf")
    monkeypatch.setattr('google.generativeai.GenerativeModel', lambda name: DummyModel())
    monkeypatch.setattr('google.generativeai.configure', lambda api_key: None)
    processor = GeminiDocumentProcessor(api_key='dummy')
    with pytest.raises(Exception):
        processor._extract_text_from_pdf_pages(str(pdf_path), 1, 1)

def test_summarize_text_with_timeout(monkeypatch):
    monkeypatch.setattr('google.generativeai.GenerativeModel', lambda name: DummyModel())
    monkeypatch.setattr('google.generativeai.configure', lambda api_key: None)
    processor = GeminiDocumentProcessor(api_key='dummy')
    summary = processor._summarize_text_with_timeout("test", 1, 1, 1)
    assert summary == "Summary"

def test_save_summaries(tmp_path, monkeypatch):
    monkeypatch.setattr('google.generativeai.GenerativeModel', lambda name: DummyModel())
    monkeypatch.setattr('google.generativeai.configure', lambda api_key: None)
    processor = GeminiDocumentProcessor(api_key='dummy')
    output = tmp_path / "out.md"
    summaries = {"Chunk 1": "Summary text"}
    images = {"Chunk 1": [{"path": "img.png", "alt": "img"}]}
    meta = {"title": "Test"}
    path = processor.save_summaries(summaries, images, str(output), "Test", "pdf", meta)
    assert os.path.exists(path)

def test_process_chunk(monkeypatch, tmp_path):
    pdf_path = tmp_path / "test.pdf"
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF")
    class DummyPage:
        def extract_text(self):
            return "Hello"
    class DummyReader:
        pages = [DummyPage(), DummyPage()]
    monkeypatch.setattr('pypdf.PdfReader', lambda f: DummyReader())
    monkeypatch.setattr('google.generativeai.GenerativeModel', lambda name: DummyModel())
    monkeypatch.setattr('google.generativeai.configure', lambda api_key: None)
    processor = GeminiDocumentProcessor(api_key='dummy')
    summary, images = processor.process_chunk(str(pdf_path), 1, 2)
    assert isinstance(summary, str)
    assert isinstance(images, list)

def test_process_chunk_no_images(monkeypatch, tmp_path):
    pdf_path = tmp_path / "test.pdf"
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF")
    class DummyPage:
        def extract_text(self):
            return "Hello"
    class DummyReader:
        pages = [DummyPage(), DummyPage()]
    monkeypatch.setattr('pypdf.PdfReader', lambda f: DummyReader())
    monkeypatch.setattr('google.generativeai.GenerativeModel', lambda name: DummyModel())
    monkeypatch.setattr('google.generativeai.configure', lambda api_key: None)
    processor = GeminiDocumentProcessor(api_key='dummy', extract_images=False)
    summary, images = processor.process_chunk(str(pdf_path), 1, 2)
    assert isinstance(summary, str)
    assert images == []

def test_retry_failed_chunks_no_failed(monkeypatch):
    monkeypatch.setattr('google.generativeai.GenerativeModel', lambda name: DummyModel())
    monkeypatch.setattr('google.generativeai.configure', lambda api_key: None)
    processor = GeminiDocumentProcessor(api_key='dummy')
    summaries = {"Chunk 1": "Summary"}
    updated = processor.retry_failed_chunks('dummy.pdf', summaries)
    assert updated == summaries
