import os
import sys
import json
import time
import logging
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Any, Optional

import pypdf
import google.generativeai as genai
from PIL import Image

# Import the epub processor
from epub_processor import SimpleEpubProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GeminiDocumentProcessor")


class ChunkTimeoutError(Exception):
    """Exception raised when a chunk processing times out."""
    pass


class GeminiDocumentProcessor:
    """
    Enhanced processor for PDF and EPUB documents using Gemini API with improved
    timeout handling and chunk-level processing.
    """

    def __init__(
            self,
            api_key: str = None,
            model_name: str = "gemini-2.0-flash",
            language: str = "thai",
            chunk_size: int = 7,
            max_retries: int = 3,
            retry_delay: int = 5,
            extract_images: bool = True,
            min_img_width: int = 100,
            min_img_height: int = 100,
            img_format: str = "png",
            max_workers: int = 4,
            request_timeout: int = 60
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.language = language
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.extract_images = extract_images
        self.min_img_width = min_img_width
        self.min_img_height = min_img_height
        self.img_format = img_format.lower()
        self.max_workers = max_workers
        self.request_timeout = request_timeout
        self.failed_chunks = []
        self.progress_callback = None  # Added progress callback
        # Do NOT initialize Gemini API here
        self.model = None

    def _initialize_api(self):
        """Initialize the Gemini API client."""
        if self.model is not None:
            return  # Already initialized
        try:
            if self.api_key:
                logger.info("Using provided API key")
                genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            # Test the API connection
            response = self.model.generate_content("Hello, this is a test.")
            if response:
                logger.info(f"Successfully connected to Gemini API using model {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing Gemini API: {str(e)}")
            raise RuntimeError(f"Failed to initialize Gemini API: {str(e)}")

    def get_total_pages(self, pdf_path: str) -> int:
        """Get the total number of pages in a PDF document."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                return len(pdf_reader.pages)
        except Exception as e:
            logger.error(f"Error counting PDF pages: {e}")
            return 0

    def process_document(self, file_path: str) -> Tuple[Dict[str, str], Dict[str, List[Dict]], str, str, Dict]:
        """
        Process a document (PDF or EPUB) and return summaries and images.

        This is a wrapper around the chunk-level processing functions that ensures
        all chunks are processed with proper error handling.

        Args:
            file_path: Path to the document

        Returns:
            Tuple of:
            - Dictionary of chunk/chapter summaries
            - Dictionary of images by chunk/chapter
            - Document name
            - Document type ("pdf" or "epub")
            - Document metadata
        """
        # Determine document type
        file_extension = os.path.splitext(file_path)[1].lower()
        doc_type = "epub" if file_extension == ".epub" else "pdf"
        doc_name = os.path.splitext(os.path.basename(file_path))[0]

        # Reset failed chunks list
        self.failed_chunks = []

        # Create images directory if extracting images
        if self.extract_images:
            images_dir = os.path.join(os.path.dirname(file_path), f"{doc_name}_images")
            os.makedirs(images_dir, exist_ok=True)
            logger.info(f"Images will be saved to: {images_dir}")

        # Process based on document type
        if doc_type == "pdf":
            return self._process_pdf(file_path)
        else:
            return self._process_epub(file_path)

    def _process_pdf(self, pdf_path: str) -> Tuple[Dict[str, str], Dict[str, List[Dict]], str, str, Dict]:
        """
        Process a PDF document chunk by chunk with immediate retry on failure.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Tuple containing:
            - Dictionary of summaries by chunk
            - Dictionary of images by chunk
            - Document name without extension
            - Document type ("pdf")
            - Document metadata
        """
        # Get document name
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

        # Count total pages
        total_pages = self.get_total_pages(pdf_path)
        logger.info(f"Total pages in PDF: {total_pages}")

        # Calculate total chunks
        total_chunks = (total_pages + self.chunk_size - 1) // self.chunk_size

        # Extract metadata
        metadata = self.extract_metadata(pdf_path)

        # Process each chunk
        summaries = {}
        images_by_chunk = {}

        # Process chunks sequentially to avoid overwhelming the API
        for chunk_num in range(1, total_chunks + 1):
            page_start = (chunk_num - 1) * self.chunk_size + 1
            page_end = min(chunk_num * self.chunk_size, total_pages)

            # Update progress via callback if it exists
            if self.progress_callback:
                self.progress_callback(chunk_num, total_chunks)

            # Immediate retry loop for each chunk
            max_immediate_retries = 2  # Try up to 2 additional times (total 3 attempts)
            immediate_retry_count = 0
            chunk_success = False
            last_error = None

            while immediate_retry_count <= max_immediate_retries and not chunk_success:
                try:
                    if immediate_retry_count > 0:
                        # Log that we're retrying this chunk immediately
                        logger.info(
                            f"Immediately retrying chunk {chunk_num} (attempt {immediate_retry_count + 1}/{max_immediate_retries + 1})...")

                        # Adjust model for retry - use a more capable model on retry
                        if immediate_retry_count == 1 and self.model_name == "gemini-2.0-flash":
                            original_model = self.model_name
                            self.model_name = "gemini-2.5-flash-preview-04-17"
                            logger.info(f"  Switching from {original_model} to {self.model_name} for retry")
                        elif immediate_retry_count == 2:
                            # On second retry, increase timeout
                            original_timeout = self.request_timeout
                            self.request_timeout = original_timeout * 1.5
                            logger.info(f"  Increasing timeout to {self.request_timeout} seconds for final retry")
                    else:
                        # First attempt
                        logger.info(f"Summarizing chunk {chunk_num} (pages {page_start}-{page_end})...")

                    # Extract text and images from this chunk
                    chunk_text = self._extract_text_from_pdf_pages(pdf_path, page_start, page_end)
                    chunk_images = []

                    if self.extract_images:
                        chunk_images = self._extract_images_from_pdf_pages(
                            pdf_path, page_start, page_end,
                            os.path.join(os.path.dirname(pdf_path), f"{pdf_name}_images"),
                            chunk_num
                        )

                    # Summarize the chunk text
                    chunk_summary = self._summarize_text_with_timeout(
                        chunk_text, chunk_num, page_start, page_end,
                        doc_type="pdf", doc_filename=pdf_name
                    )

                    # Save temporary summary
                    temp_file = f"temp_{pdf_name}_chunk_{chunk_num}.md"
                    with open(temp_file, "w", encoding="utf-8") as f:
                        f.write(chunk_summary)
                    logger.info(f"  Saved temporary summary to {temp_file}")

                    # Store results
                    summaries[f"Chunk {chunk_num}"] = chunk_summary

                    if chunk_images:
                        images_by_chunk[f"Chunk {chunk_num}"] = chunk_images
                        logger.info(f"  Associated {len(chunk_images)} images with chunk {chunk_num}")

                    # If we get here, the chunk was processed successfully
                    chunk_success = True

                    # Reset any modified parameters
                    if immediate_retry_count > 0:
                        if immediate_retry_count == 1 and 'original_model' in locals():
                            logger.info(f"  Resetting model back to {original_model}")
                            self.model_name = original_model
                        if immediate_retry_count == 2 and 'original_timeout' in locals():
                            logger.info(f"  Resetting timeout back to {original_timeout}")
                            self.request_timeout = original_timeout

                except Exception as e:
                    last_error = e
                    logger.error(f"Error processing chunk {chunk_num} (pages {page_start}-{page_end}): {str(e)}")
                    immediate_retry_count += 1

                    # Reset any modified parameters before next retry or moving on
                    if 'original_model' in locals() and self.model_name != original_model:
                        logger.info(f"  Resetting model back to {original_model}")
                        self.model_name = original_model
                    if 'original_timeout' in locals() and self.request_timeout != original_timeout:
                        logger.info(f"  Resetting timeout back to {original_timeout}")

            # After all retry attempts, if still not successful, add to failed chunks
            if not chunk_success:
                self.failed_chunks.append({
                    "chunk_number": chunk_num,
                    "page_start": page_start,
                    "page_end": page_end,
                    "error": str(last_error)
                })

                # Create a placeholder summary for the failed chunk
                summaries[
                    f"Chunk {chunk_num}"] = f"**Error processing pages {page_start}-{page_end} after {max_immediate_retries + 1} attempts:** {str(last_error)}"

        # Final progress update
        if self.progress_callback:
            self.progress_callback(total_chunks, total_chunks)

        return summaries, images_by_chunk, pdf_name, "pdf", metadata

    def _extract_text_from_pdf_pages(self, pdf_path: str, start_page: int, end_page: int) -> str:
        """
        Extract text from a range of PDF pages.

        Args:
            pdf_path: Path to the PDF file
            start_page: Start page number (1-based index)
            end_page: End page number (inclusive)

        Returns:
            Extracted text from the specified pages
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)

                # Adjust for 0-based indexing
                start_idx = start_page - 1
                end_idx = min(end_page, len(pdf_reader.pages)) - 1

                text_parts = []
                for i in range(start_idx, end_idx + 1):
                    try:
                        page = pdf_reader.pages[i]
                        page_text = page.extract_text() or ""
                        text_parts.append(f"--- Page {i + 1} ---\n{page_text}")
                    except Exception as e:
                        logger.error(f"Error extracting text from page {i + 1}: {str(e)}")
                        text_parts.append(f"--- Page {i + 1} ---\n[Error extracting text: {str(e)}]")

                return "\n\n".join(text_parts)

        except Exception as e:
            logger.error(f"Error opening PDF file: {str(e)}")
            raise

    def _extract_images_from_pdf_pages(
            self, pdf_path: str, start_page: int, end_page: int,
            output_dir: str, chunk_num: int = None) -> List[Dict]:
        """
        Extract images from a range of PDF pages using PyPDF.

        Args:
            pdf_path: Path to the PDF file
            start_page: Start page number (1-based index)
            end_page: End page number (inclusive)
            output_dir: Directory to save extracted images
            chunk_num: Current chunk number (for logging)

        Returns:
            List of dictionaries with image information
        """
        # Store information about extracted images
        extracted_images = []

        try:
            # Make sure output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Let's try with PyPDF first - it's more widely available
            # For each page in range, extract images if any
            with open(pdf_path, "rb") as pdf_file:
                pdf = pypdf.PdfReader(pdf_file)

                # Adjust for 0-based indexing
                start_idx = start_page - 1
                end_idx = min(end_page, len(pdf.pages)) - 1

                # Process each page in range
                for page_idx in range(start_idx, end_idx + 1):
                    page_num = page_idx + 1  # 1-based page number for display
                    page = pdf.pages[page_idx]

                    # Try to get images from the page (if any)
                    try:
                        # Extract image data through page.images property
                        if hasattr(page, "images") and page.images:
                            images_extracted = 0

                            for i, image in enumerate(page.images):
                                try:
                                    # Skip small images
                                    if hasattr(image, "width") and hasattr(image, "height"):
                                        if image.width < self.min_img_width or image.height < self.min_img_height:
                                            continue

                                    # Create a unique filename for this image
                                    img_filename = f"page{page_num:03d}_img{i + 1:03d}.{self.img_format}"
                                    img_path = os.path.join(output_dir, img_filename)

                                    # Try to save the image
                                    with open(img_path, "wb") as img_file:
                                        img_file.write(image.data)

                                    # Check if image was saved successfully
                                    if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                                        # Store image information
                                        extracted_images.append({
                                            "filename": img_filename,
                                            "path": img_path,
                                            "page": page_num,
                                            "width": getattr(image, "width", 0),
                                            "height": getattr(image, "height", 0),
                                            "alt": f"Image {i + 1} from page {page_num}"
                                        })
                                        images_extracted += 1

                                except Exception as e:
                                    logger.error(f"Error extracting image {i + 1} from page {page_num}: {str(e)}")

                            if images_extracted > 0:
                                logger.info(f"Extracted {images_extracted} images from page {page_num}")

                    except Exception as e:
                        logger.error(f"Error accessing images on page {page_num}: {str(e)}")
                        continue

        except Exception as e:
            logger.error(f"Error in image extraction process: {str(e)}")

        # Return information about extracted images
        return extracted_images

    def _summarize_text_with_timeout(
            self, text: str, chunk_num: int, page_start: int, page_end: int,
            doc_type: str = "pdf", doc_filename: str = "") -> str:
        """
        Summarize text using Gemini API with timeout handling.

        Args:
            text: Text to summarize
            chunk_num: Chunk number for error reporting
            page_start: Start page of this chunk
            page_end: End page of this chunk
            doc_type: Document type ('pdf' or 'epub')
            doc_filename: Name of the document file

        Returns:
            Summarized text

        Raises:
            ChunkTimeoutError: If the API call times out
        """
        start_time = time.time()
        max_time = self.request_timeout  # seconds

        # Ensure Gemini API is initialized before summarization
        self._initialize_api()

        # Prepare the original Thai prompt
        page_or_chapter = "หน้า" if doc_type == 'pdf' else "บท"
        prompt = (
            "คุณคือ expert ด้าน summarizer analyzing\n\n"
            f"ช่วยสรุปเนื้อหา**จาก{doc_type}** ({doc_filename}) **เป็น ภาษาไทย** โดย:\n"
            "1. ใช้หัวข้อเดิมตามไฟล์ได้เลย (ไม่ต้องแปลหัวข้อ)\n"
            "2. ไม่อยากให้ตกหล่นแม้แต่เรื่องเดียว (ขอแบบละเอียดจนไม่ต้องกลับไปอ่านต้นฉบับเลย)\n"
            f"3. บอกด้วยว่ากำลังสรุป{page_or_chapter}ไหนของไฟล์ เช่น <!-- 1 -->, <!-- 2 -->, <!-- 3 -->, <!-- end --> (use markdown comment)\n"
            "4. ไม่จำเป็นต้องกระชับ และรักษาความถูกต้องของข้อมูลสำคัญ เนื้อหาสำคัญไม่ตกหล่น\n"
            "5. ไม่ต้องแปล technical terminology จากภาษาอังกฤษให้เป็นภาษาไทย\n"
            "6. ถ้าใน file มีตัวอย่าง code ก็ใส่มาให้ด้วย\n"
            "7. output ใน format ที่ดีที่สุด\n\n"
            f"เนื้อหาต่อไปนี้มาจาก{page_or_chapter} {page_start} ถึง{page_or_chapter} {page_end}:\n\n{text}"
        )

        # Define a timeout function using a separate thread
        def call_with_timeout(func, *args, **kwargs):
            result = [None]
            error = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    error[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(max_time)

            if thread.is_alive():
                # Thread is still running after timeout
                raise ChunkTimeoutError(f"API call timed out after {max_time} seconds")

            if error[0]:
                raise error[0]

            return result[0]

        # Try multiple times with backoff
        for attempt in range(1, self.max_retries + 1):
            try:
                # Check if we've already exceeded our timeout
                elapsed = time.time() - start_time
                if elapsed > max_time:
                    raise ChunkTimeoutError(f"Processing timeout after {elapsed:.1f} seconds")

                # Set up the generation config (without timeout parameter)
                generation_config = {
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "top_k": 50,
                    "max_output_tokens": 65000,
                    "response_mime_type": "text/plain"
                }

                # Make the API call with our own timeout handling
                # Using only valid roles (user)
                response = call_with_timeout(
                    self.model.generate_content,
                    prompt,  # Send the full prompt as a user message
                    generation_config=generation_config
                )

                # Successfully got a response
                return response.text

            except ChunkTimeoutError:
                # Re-raise timeout errors directly
                raise

            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Attempt {attempt} failed for chunk {chunk_num}: {error_msg}")

                if attempt < self.max_retries:
                    # Wait before retrying with exponential backoff
                    wait_time = self.retry_delay * (2 ** (attempt - 1))
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    # Failed all attempts
                    logger.error(f"All {self.max_retries} attempts failed for chunk {chunk_num}")
                    raise RuntimeError(f"Failed to summarize after {self.max_retries} attempts: {error_msg}")

    def _process_epub(self, epub_path: str) -> Tuple[Dict[str, str], Dict[str, List[Dict]], str, str, Dict]:
        """
        Process an EPUB document chapter by chapter.

        Args:
            epub_path: Path to the EPUB file

        Returns:
            Tuple containing:
            - Dictionary of summaries by chapter
            - Dictionary of images by chapter
            - Document name without extension
            - Document type ("epub")
            - Document metadata
        """
        # Create EPUB processor
        epub_processor = SimpleEpubProcessor(extract_images=self.extract_images, img_format=self.img_format)

        # Extract chapters and metadata
        chapters, images_by_chapter, epub_name, metadata = epub_processor.process_epub(
            epub_path,
            image_output_dir=os.path.join(os.path.dirname(epub_path),
                                          f"{os.path.splitext(os.path.basename(epub_path))[0]}_images")
        )

        # Process each chapter with timeout handling
        summaries = {}
        processed_images = {}

        # Calculate total chapters
        total_chapters = len(chapters)

        # Initialize progress
        chapter_count = 0

        for chapter_key, chapter_text in chapters.items():
            chapter_count += 1

            # Update progress via callback if it exists
            if self.progress_callback:
                self.progress_callback(chapter_count, total_chapters)

            try:
                logger.info(f"Summarizing {chapter_key}...")

                # Extract chapter number if possible
                chapter_num = chapter_key.replace("Chapter ", "")

                # Summarize this chapter with timeout handling
                chapter_summary = self._summarize_text_with_timeout(
                    chapter_text,
                    chapter_num,
                    chapter_num, chapter_num,  # Using chapter number for both start and end
                    doc_type="epub",
                    doc_filename=epub_name
                )

                # Save temporary summary
                temp_file = f"temp_{epub_name}_{chapter_key.lower().replace(' ', '_')}.md"
                with open(temp_file, "w", encoding="utf-8") as f:
                    f.write(chapter_summary)
                logger.info(f"  Saved temporary summary to {temp_file}")

                # Store results
                summaries[chapter_key] = chapter_summary

                # Store images for this chapter if any
                if chapter_key in images_by_chapter:
                    processed_images[chapter_key] = images_by_chapter[chapter_key]
                    logger.info(f"  Associated {len(images_by_chapter[chapter_key])} images with {chapter_key}")

            except Exception as e:
                logger.error(f"Error processing {chapter_key}: {str(e)}")

                # Record the failure
                self.failed_chunks.append({
                    "chapter": chapter_key,
                    "error": str(e)
                })

                # Create a placeholder summary for the failed chapter
                summaries[chapter_key] = f"**Error processing {chapter_key}:** {str(e)}"

                # Continue with the next chapter

        # Final progress update
        if self.progress_callback:
            self.progress_callback(total_chapters, total_chapters)

        return summaries, processed_images, epub_name, "epub", metadata
    def extract_metadata(self, file_path: str) -> Dict[str, str]:
        """
        Extract metadata from a document.

        Args:
            file_path: Path to the document

        Returns:
            Dictionary of metadata
        """
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == ".pdf":
            return self._extract_pdf_metadata(file_path)
        elif file_extension == ".epub":
            return self._extract_epub_metadata(file_path)
        else:
            return {"title": os.path.splitext(os.path.basename(file_path))[0]}

    def _extract_pdf_metadata(self, pdf_path: str) -> Dict[str, str]:
        """Extract metadata from a PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                metadata = {}

                if pdf_reader.metadata:
                    for key in pdf_reader.metadata:
                        # Clean up the key name
                        clean_key = key.lower().replace('/', '').strip()
                        if pdf_reader.metadata[key]:
                            metadata[clean_key] = str(pdf_reader.metadata[key])

                # If no title is found, use the filename
                if 'title' not in metadata or not metadata['title']:
                    metadata['title'] = os.path.splitext(os.path.basename(pdf_path))[0]

                return metadata

        except Exception as e:
            logger.error(f"Error extracting PDF metadata: {str(e)}")
            return {"title": os.path.splitext(os.path.basename(pdf_path))[0]}

    def _extract_epub_metadata(self, epub_path: str) -> Dict[str, str]:
        """Extract metadata from an EPUB file."""
        # This would ideally call a method from your epub_processor
        # For now, returning basic info
        return {"title": os.path.splitext(os.path.basename(epub_path))[0]}

    def save_summaries(
            self,
            summaries: Dict[str, str],
            images_by_chunk: Dict[str, List[Dict]],
            output_path: str,
            doc_name: str,
            doc_type: str,
            metadata: Dict,
            obsidian_metadata: Optional[Dict] = None
    ) -> str:
        """
        Save summaries to a Markdown file.

        Args:
            summaries: Dictionary of summaries by chunk/chapter
            images_by_chunk: Dictionary of images by chunk/chapter
            output_path: Path to save the output Markdown file
            doc_name: Document name
            doc_type: Document type ("pdf" or "epub")
            metadata: Document metadata
            obsidian_metadata: Optional metadata for Obsidian

        Returns:
            Path to the saved file
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                # Add YAML frontmatter for Obsidian if provided
                if obsidian_metadata:
                    f.write("---\n")

                    # Add tags
                    if 'tags' in obsidian_metadata and obsidian_metadata['tags']:
                        tags = obsidian_metadata['tags'].split(',')
                        f.write("tags:\n")
                        for tag in tags:
                            f.write(f"  - {tag.strip()}\n")

                    # Add other metadata fields
                    for key, value in obsidian_metadata.items():
                        if key != 'tags' and value:  # Skip tags as we handled them above
                            f.write(f"{key}: {value}\n")

                    f.write("---\n\n")

                # Add title and metadata
                title = metadata.get('title', doc_name)
                f.write(f"# {title}\n\n")

                if 'author' in metadata and metadata['author']:
                    f.write(f"**Author:** {metadata['author']}\n\n")

                # Add other metadata
                f.write("## Document Information\n\n")
                f.write(f"- **Type:** {doc_type.upper()}\n")

                for key, value in metadata.items():
                    if key not in ['title', 'author'] and value:
                        key_display = key.replace('_', ' ').title()
                        f.write(f"- **{key_display}:** {value}\n")

                f.write("\n")

                # Add table of contents
                f.write("## Table of Contents\n\n")

                for idx, chunk_key in enumerate(summaries.keys()):
                    f.write(f"{idx + 1}. [{chunk_key}](#{chunk_key.lower().replace(' ', '-')})\n")

                f.write("\n")

                # Add summaries
                f.write("## Summary\n\n")

                for chunk_key, summary in summaries.items():
                    # Add chunk header
                    f.write(f"### {chunk_key}\n\n")

                    # Add summary content
                    f.write(f"{summary}\n\n")

                    # Add images for this chunk if any
                    if chunk_key in images_by_chunk and images_by_chunk[chunk_key]:
                        f.write("#### Images\n\n")

                        for img_info in images_by_chunk[chunk_key]:
                            img_path = img_info.get('path', '')
                            alt_text = img_info.get('alt', 'Image') or f"Image from {chunk_key}"

                            # Create relative path for embedding
                            rel_path = os.path.relpath(
                                img_path,
                                os.path.dirname(output_path)
                            )

                            # Add image with markdown
                            f.write(f"![{alt_text}]({rel_path})\n\n")

                # Add footer with generation info
                f.write("---\n")
                f.write(f"*Summary generated using Gemini {self.model_name}*\n")

                return output_path

        except Exception as e:
            logger.error(f"Error saving summaries: {str(e)}")
            raise

    def process_chunk(self, file_path: str, page_start: int, page_end: int, timeout: int = 60) -> Tuple[
        str, List[Dict]]:
        """
        Process a single chunk of a document with timeout.

        Args:
            file_path: Path to the document
            page_start: Start page number (1-based index)
            page_end: End page number (inclusive)
            timeout: Timeout in seconds

        Returns:
            Tuple containing:
            - Summarized text
            - List of image information dictionaries
        """
        # Save the current timeout setting
        original_timeout = self.request_timeout

        try:
            # Update timeout for this call
            self.request_timeout = timeout

            # Get document info
            doc_type = "pdf" if file_path.lower().endswith('.pdf') else "epub"
            doc_filename = os.path.basename(file_path)

            # Extract text from pages
            text = self._extract_text_from_pdf_pages(file_path, page_start, page_end)

            # Summarize text with timeout
            summary = self._summarize_text_with_timeout(
                text,
                0,  # Not using chunk number here
                page_start,
                page_end,
                doc_type=doc_type,
                doc_filename=doc_filename
            )

            # Extract images if enabled
            images = []
            if self.extract_images:
                doc_name = os.path.splitext(os.path.basename(file_path))[0]
                images_dir = os.path.join(os.path.dirname(file_path), f"{doc_name}_images")

                images = self._extract_images_from_pdf_pages(
                    file_path, page_start, page_end, images_dir
                )

            return summary, images

        finally:
            # Restore original timeout
            self.request_timeout = original_timeout

    def retry_failed_chunks(self, file_path: str, existing_summaries: Dict[str, str]) -> Dict[str, str]:
        """
        Retry processing previously failed chunks.

        Args:
            file_path: Path to the document
            existing_summaries: Dictionary of existing summaries

        Returns:
            Updated dictionary of summaries
        """
        # Copy existing summaries
        updated_summaries = existing_summaries.copy()

        # Nothing to do if no failed chunks
        if not self.failed_chunks:
            return updated_summaries

        logger.info(f"Retrying {len(self.failed_chunks)} failed chunks with increased timeout")

        # Save original settings
        original_timeout = self.request_timeout
        original_retries = self.max_retries

        try:
            # Use more aggressive settings for retries
            self.request_timeout = original_timeout * 2
            self.max_retries = original_retries + 2

            # Process each failed chunk
            for failed_chunk in self.failed_chunks:
                chunk_num = failed_chunk.get("chunk_number")
                if not chunk_num:
                    continue

                page_start = failed_chunk.get("page_start", 0)
                page_end = failed_chunk.get("page_end", 0)

                if page_start <= 0 or page_end <= 0:
                    continue

                try:
                    logger.info(f"Retrying chunk {chunk_num} (pages {page_start}-{page_end})...")

                    # Process this chunk
                    summary, _ = self.process_chunk(file_path, page_start, page_end)

                    # Update the summaries
                    chunk_key = f"Chunk {chunk_num}"
                    updated_summaries[chunk_key] = summary

                    # Remove from failed chunks
                    logger.info(f"Successfully reprocessed chunk {chunk_num}")

                except Exception as e:
                    logger.error(f"Failed to reprocess chunk {chunk_num}: {str(e)}")
                    # Keep the existing summary or error message

            return updated_summaries

        finally:
            # Restore original settings
            self.request_timeout = original_timeout
            self.max_retries = original_retries