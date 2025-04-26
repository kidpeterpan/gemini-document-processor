import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import os
import logging
import re
import html2text

# Set up logging
logger = logging.getLogger("GeminiEbookProcessor")


class SimpleEpubProcessor:
    """
    A simplified EPUB processor that doesn't require PIL for image processing.
    """

    def __init__(self, extract_images=True, img_format="png"):
        self.extract_images = extract_images
        self.img_format = img_format.lower()
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = False
        self.html_converter.ignore_tables = False
        self.html_converter.body_width = 0  # No wrapping

    def process_epub(self, epub_path, image_output_dir=None):
        """
        Process an EPUB file and extract text and images.

        Args:
            epub_path (str): Path to the EPUB file.
            image_output_dir (str, optional): Directory to save extracted images.
                                            If None, a temp directory will be created.

        Returns:
            tuple: (chapters, images_by_chapter, epub_name_without_ext, metadata)
                chapters: Dictionary of chapter number to text content
                images_by_chapter: Dictionary of chapter number to list of image information
                epub_name_without_ext: Name of the EPUB file without extension
                metadata: Dictionary of metadata extracted from the EPUB
        """
        try:
            if not os.path.exists(epub_path):
                raise FileNotFoundError(f"EPUB file not found: {epub_path}")

            # Get EPUB filename for use in the output
            epub_filename = os.path.basename(epub_path)
            epub_name_without_ext = os.path.splitext(epub_filename)[0]

            # Create image output directory if extracting images
            if self.extract_images and image_output_dir:
                os.makedirs(image_output_dir, exist_ok=True)
                logger.info(f"Images will be saved to: {image_output_dir}")

            # Read the EPUB file
            book = epub.read_epub(epub_path)

            # Extract metadata
            metadata = {}
            try:
                for key, value in book.get_metadata('DC', 'http://purl.org/dc/elements/1.1/').items():
                    if value:
                        # Strip namespace from key
                        key = key.split("}")[-1]
                        metadata[key] = value[0][0]
            except Exception as e:
                logger.error(f"Error extracting metadata: {str(e)}")
                metadata = {"title": epub_name_without_ext}

            # Get chapters and images
            chapters = {}  # Always initialize as a dictionary
            images_by_chapter = {}
            chapter_index = 1

            try:
                for item in book.get_items():
                    if item.get_type() == ebooklib.ITEM_DOCUMENT:
                        try:
                            # Extract text from HTML content
                            content = item.get_content().decode('utf-8')
                            soup = BeautifulSoup(content, 'html.parser')

                            # Get text content
                            text_content = self.html_converter.handle(content)

                            # Skip if there's no meaningful content
                            if not self._has_content(text_content):
                                continue

                            # Add chapter
                            chapter_key = f"Chapter {chapter_index}"
                            chapters[chapter_key] = text_content

                            # Basic image extraction that doesn't depend on PIL
                            if self.extract_images and image_output_dir:
                                chapter_images = self._extract_images_basic(
                                    book, item, soup, image_output_dir,
                                    epub_name_without_ext, chapter_index
                                )

                                if chapter_images:
                                    images_by_chapter[chapter_key] = chapter_images
                                    logger.info(f"Found {len(chapter_images)} image references in {chapter_key}")

                            chapter_index += 1
                        except Exception as e:
                            logger.error(f"Error processing chapter {chapter_index}: {str(e)}")
                            # Continue with next chapter
                            chapter_index += 1
            except Exception as e:
                logger.error(f"Error processing chapters: {str(e)}")
                # If we hit an error in the chapter processing loop, add an error chapter
                chapters = {"Error": f"Failed to process chapters: {str(e)}"}

            # If no chapters were found, add a placeholder
            if not chapters:
                chapters = {"No Content": "No readable content found in the EPUB file."}

            return chapters, images_by_chapter, epub_name_without_ext, metadata

        except Exception as e:
            logger.error(f"Error processing EPUB file: {str(e)}")
            # Return minimal data to not break the pipeline - ensure everything is a dictionary
            return {"Error": f"Failed to process EPUB: {str(e)}"}, {}, epub_name_without_ext, {
                "title": epub_name_without_ext}

    def _has_content(self, text):
        """Check if there's substantial content in the text."""
        # Remove whitespace, special characters, and common EPUB navigation elements
        if not text:
            return False
        clean_text = re.sub(r'\s+|[^\w]', '', text)
        return len(clean_text) > 100  # Arbitrary threshold

    def _extract_images_basic(self, book, item, soup, output_dir, epub_name, chapter_index):
        """Extract basic image references without using PIL."""
        extracted_images = []
        img_elements = soup.find_all('img')

        for img_index, img in enumerate(img_elements):
            try:
                # Get image source
                src = img.get('src')
                if not src:
                    continue

                # Just record the image reference without saving it
                # This way we don't depend on PIL but still have image info
                img_filename = f"{epub_name}_chapter{chapter_index:03d}_img{img_index + 1:03d}.{self.img_format}"
                img_path = os.path.join(output_dir, img_filename)

                # Store image information
                extracted_images.append({
                    'filename': img_filename,
                    'path': img_path,
                    'chapter': chapter_index,
                    'alt': img.get('alt', ''),
                    'src': src
                })

            except Exception as e:
                logger.error(f"Error extracting image {img_index} reference from chapter {chapter_index}: {str(e)}")
                continue

        return extracted_images