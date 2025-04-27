import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from epub_processor import SimpleEpubProcessor

class DummyBook:
    def get_metadata(self, *args, **kwargs):
        return {'title': [('Test Book',)]}
    def get_items(self):
        class DummyItem:
            def get_type(self): return 9  # ebooklib.ITEM_DOCUMENT
            def get_content(self): return b'<html><body><h1>Chapter</h1><p>Content</p><img src="img.png"/></body></html>'
        return [DummyItem()]

def test_process_epub(monkeypatch, tmp_path):
    epub_path = tmp_path / "test.epub"
    with open(epub_path, "wb") as f:
        f.write(b"dummy epub content")
    monkeypatch.setattr('ebooklib.epub.read_epub', lambda path: DummyBook())
    processor = SimpleEpubProcessor(extract_images=True)
    chapters, images_by_chapter, name, meta = processor.process_epub(str(epub_path), image_output_dir=str(tmp_path))
    assert isinstance(chapters, dict)
    assert isinstance(images_by_chapter, dict)
    assert name == "test"
    assert meta['title'] == 'Test Book'

def test_process_epub_missing_file(tmp_path):
    processor = SimpleEpubProcessor()
    chapters, images_by_chapter, name, meta = processor.process_epub(str(tmp_path / "no_file.epub"))
    assert "Error" in chapters or "Failed" in list(chapters.values())[0]

def test_process_epub_no_chapters(monkeypatch, tmp_path):
    class DummyBook:
        def get_metadata(self, *args, **kwargs):
            return {'title': [('Test Book',)]}
        def get_items(self):
            return []
    epub_path = tmp_path / "test.epub"
    with open(epub_path, "wb") as f:
        f.write(b"dummy epub content")
    monkeypatch.setattr('ebooklib.epub.read_epub', lambda path: DummyBook())
    processor = SimpleEpubProcessor()
    chapters, images_by_chapter, name, meta = processor.process_epub(str(epub_path), image_output_dir=str(tmp_path))
    assert "No Content" in chapters

def test_has_content():
    processor = SimpleEpubProcessor()
    assert not processor._has_content("")
    assert not processor._has_content("\n\t ")
    assert processor._has_content("a" * 101)

def test_extract_images_basic(tmp_path):
    processor = SimpleEpubProcessor()
    class DummyBook: pass
    class DummyItem: pass
    from bs4 import BeautifulSoup
    soup = BeautifulSoup('<img src="img1.png" alt="A"/><img src="img2.png"/>', 'html.parser')
    images = processor._extract_images_basic(DummyBook(), DummyItem(), soup, str(tmp_path), "book", 1)
    assert len(images) == 2
    assert images[0]['filename'].endswith('.png')
    assert images[0]['alt'] == 'A'

def test_extract_images_basic_malformed_html(tmp_path):
    processor = SimpleEpubProcessor()
    class DummyBook: pass
    class DummyItem: pass
    from bs4 import BeautifulSoup
    soup = BeautifulSoup('<img><div><img src=""></div>', 'html.parser')
    images = processor._extract_images_basic(DummyBook(), DummyItem(), soup, str(tmp_path), "book", 1)
    assert isinstance(images, list)

def test_has_content_special_chars():
    processor = SimpleEpubProcessor()
    assert not processor._has_content("!@#$%^&*()_+-=\n\t ")
