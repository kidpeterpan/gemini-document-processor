import os
import sys
import json
import threading
import logging
import shutil
import time
import uuid
import signal
from pathlib import Path
from threading import Timer
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory

# Import the enhanced document processor
from document_processor import GeminiDocumentProcessor, ChunkTimeoutError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("document_processor_web.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GeminiDocumentProcessorWeb")

app = Flask(__name__)

# Store processing jobs
jobs = {}
uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
obsidian_dir = None  # Will be set from settings or form

# Create directories if they don't exist
os.makedirs(uploads_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Try to load obsidian_dir from settings
try:
    settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.json')
    if os.path.exists(settings_file):
        with open(settings_file, 'r') as f:
            settings = json.load(f)
            obsidian_dir = settings.get('obsidian_vault_path', None)
            if obsidian_dir and os.path.exists(obsidian_dir):
                logger.info(f"Loaded Obsidian vault path from settings: {obsidian_dir}")
            else:
                obsidian_dir = None
                logger.warning("Obsidian vault path not found or invalid in settings")
except Exception as e:
    logger.error(f"Error loading settings: {e}")


class ProcessingJob:
    def __init__(self, job_id, file_path, output_path, settings):
        self.job_id = job_id
        self.file_path = file_path
        self.output_path = output_path
        self.settings = settings
        self.status = "pending"
        self.progress = 0
        self.message = "Waiting to start..."
        self.log = []
        self.result_files = []
        self.error = None
        self.thread = None
        self.document_type = None
        self.document_metadata = None
        self.current_chunk = None
        self.total_chunks = None
        self.chunk_start_time = None
        self.chunk_timer = None
        self.processing_paused = False
        self.failed_chunks = []
        self.retry_for_job_id = None  # For retry jobs, reference to original job

    def add_log(self, message):
        timestamp = time.strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        self.log.append(log_entry)
        self.message = message
        logger.info(f"Job {self.job_id}: {message}")

    def start_chunk_timer(self, chunk_number, timeout_seconds=300):  # 5-minute default timeout
        """Start a timer for the current chunk being processed."""
        self.current_chunk = chunk_number
        self.chunk_start_time = time.time()

        # Cancel any existing timer
        if self.chunk_timer:
            self.chunk_timer.cancel()

        # Set a new timer
        self.chunk_timer = Timer(timeout_seconds, self.handle_chunk_timeout)
        self.chunk_timer.daemon = True
        self.chunk_timer.start()

    def handle_chunk_timeout(self):
        """Handle a chunk processing timeout by logging and adding to failed chunks."""
        elapsed = time.time() - self.chunk_start_time
        self.add_log(f"⚠️ WARNING: Chunk {self.current_chunk} processing timed out after {elapsed:.1f} seconds")

        # Add to failed chunks list
        if self.current_chunk is not None:
            self.failed_chunks.append({
                "chunk_number": self.current_chunk,
                "reason": "timeout"
            })

        # Set the job as paused so we can try to recover
        self.processing_paused = True

    def clear_chunk_timer(self):
        """Clear the chunk timer after successful completion."""
        if self.chunk_timer:
            self.chunk_timer.cancel()
            self.chunk_timer = None
        self.chunk_start_time = None

    def update_progress(self):
        """Update progress percentage based on chunks processed."""
        if self.total_chunks and self.current_chunk is not None:
            # Ensure current_chunk is not greater than total_chunks
            current = min(self.current_chunk, self.total_chunks)
            # Calculate progress as percentage of chunks completed
            self.progress = min(round((current / self.total_chunks) * 100), 99)
            # Only set to 100% when fully complete
            if self.status == "completed":
                self.progress = 100

            # Log progress updates for debugging
            logger.info(
                f"Job {self.job_id}: Progress updated to {self.progress}% (chunk {current}/{self.total_chunks})")

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "log": self.log[-15:] if len(self.log) > 15 else self.log,  # Return last 15 log entries
            "result_files": self.result_files,
            "error": self.error,
            "document_type": self.document_type,
            "document_metadata": self.document_metadata,
            "current_chunk": self.current_chunk,
            "total_chunks": self.total_chunks,
            "failed_chunks": len(self.failed_chunks) if self.failed_chunks else 0
        }


def capture_output(job):
    """Redirect stdout to capture logs for a job"""

    class OutputCapture:
        def __init__(self, job):
            self.job = job
            self.original_stdout = sys.stdout
            self.buffer = ""

        def write(self, message):
            self.original_stdout.write(message)

            # Add to buffer
            self.buffer += message

            # Check if we have a complete line
            if "\n" in self.buffer:
                lines = self.buffer.split("\n")
                # Keep the last incomplete line in the buffer
                self.buffer = lines[-1]

                # Log complete lines
                for line in lines[:-1]:
                    if line.strip():  # Skip empty lines
                        self.job.add_log(line.strip())

        def flush(self):
            self.original_stdout.flush()

            # Flush any remaining content in the buffer
            if self.buffer.strip():
                self.job.add_log(self.buffer.strip())
                self.buffer = ""

    return OutputCapture(job)


def save_to_obsidian(file_path, obsidian_dir, job, settings=None):
    """
    Save a file to the Obsidian vault.

    Args:
        file_path (str): Path to the source file
        obsidian_dir (str): Path to the Obsidian vault
        job (ProcessingJob): The job object for logging
        settings (dict, optional): Additional settings

    Returns:
        str: Path to the file in the Obsidian vault
    """
    if not obsidian_dir or not os.path.exists(obsidian_dir):
        job.add_log("❌ Invalid Obsidian vault path")
        return None

    try:
        # Create target directory (books folder in Obsidian vault)
        file_name = os.path.basename(file_path)
        books_dir = os.path.join(obsidian_dir, "books")
        os.makedirs(books_dir, exist_ok=True)

        # Target path in Obsidian vault
        target_path = os.path.join(books_dir, file_name)

        # Copy file to Obsidian vault
        shutil.copy2(file_path, target_path)

        # Copy associated images directory if it exists
        if job.document_type:
            base_name = os.path.splitext(os.path.basename(job.file_path))[0]
            images_dir = os.path.join(os.path.dirname(job.file_path), f"{base_name}_images")
            if os.path.exists(images_dir):
                target_images_dir = os.path.join(books_dir, f"{base_name}_images")

                # Remove existing images directory if it exists
                if os.path.exists(target_images_dir):
                    shutil.rmtree(target_images_dir)

                # Copy images directory
                shutil.copytree(images_dir, target_images_dir)
                job.add_log(f"✅ Copied images to Obsidian: {target_images_dir}")

        job.add_log(f"✅ File saved to Obsidian: {target_path}")
        return target_path

    except Exception as e:
        job.add_log(f"❌ Error saving to Obsidian: {str(e)}")
        return None


def process_document_job(job_id):
    """Process document in a separate thread with improved error handling and recovery"""
    job = jobs[job_id]
    job.status = "processing"
    file_extension = os.path.splitext(job.file_path)[1].lower()
    file_type = "EPUB" if file_extension == ".epub" else "PDF"
    job.add_log(f"Starting {file_type} processing for {os.path.basename(job.file_path)}")

    # Redirect stdout to capture logs
    original_stdout = sys.stdout
    sys.stdout = capture_output(job)

    try:
        # Apply settings
        settings = job.settings
        chunk_size = int(settings.get('chunk_size', 7))
        chunk_timeout = int(settings.get('chunk_timeout', 300))  # Default 5 minutes
        api_timeout = int(settings.get('api_timeout', 60))  # Default 1 minute

        # Initialize processor
        processor = GeminiDocumentProcessor(
            chunk_size=chunk_size,
            api_key=settings.get('api_key', ""),
            model_name=settings.get('model_name', "gemini-2.0-flash"),
            language="thai",  # Always Thai for this tool
            max_retries=int(settings.get('max_retries', 3)),
            extract_images=settings.get('extract_images', True),
            min_img_width=int(settings.get('min_img_width', 100)),
            min_img_height=int(settings.get('min_img_height', 100)),
            img_format=settings.get('img_format', "png"),
            max_workers=int(settings.get('max_workers', 4)),
            request_timeout=api_timeout
        )

        # Define a progress callback function
        def progress_callback(chunk_num, total_chunks):
            job.current_chunk = chunk_num
            job.total_chunks = total_chunks
            job.update_progress()

        # Add the progress callback to the processor
        processor.progress_callback = progress_callback

        # Process the document with improved chunk handling
        job.add_log(f"Processing {file_type} with {settings.get('model_name')} model")

        # First, analyze the document to get total pages/chunks
        if file_type == "PDF":
            try:
                # For PDF, we can estimate total chunks before processing
                total_pages = processor.get_total_pages(job.file_path)
                job.total_chunks = (total_pages + chunk_size - 1) // chunk_size  # Ceiling division
                job.add_log(
                    f"Document has {total_pages} pages, will process in approximately {job.total_chunks} chunks")
                # Initialize progress
                job.current_chunk = 0
                job.update_progress()
            except Exception as e:
                job.add_log(f"Error estimating document size: {str(e)}")
                # Continue anyway, we'll handle progress differently

        # Now process the document
        try:
            # Process document with standard approach and let the processor handle chunk-by-chunk processing
            summaries, images_by_chunk, doc_name, doc_type, doc_metadata = processor.process_document(job.file_path)

            # Store document type and metadata
            job.document_type = doc_type
            job.document_metadata = doc_metadata

            # Save any info about failed chunks
            job.failed_chunks = processor.failed_chunks.copy() if hasattr(processor, 'failed_chunks') else []
        except Exception as e:
            job.add_log(f"❌ Error during document processing: {str(e)}")
            raise

        # Prepare Obsidian metadata if enabled
        obsidian_metadata = None
        if settings.get('use_obsidian', False):
            obsidian_metadata = {
                'tags': settings.get('obsidian_tags', 'book,main'),
                'author': settings.get('obsidian_author', doc_metadata.get('author', '')),
                'coverUrl': settings.get('obsidian_cover_url', ''),
                'review': settings.get('obsidian_review', '')
            }

        # Save the results
        job.add_log(f"Saving summary to {job.output_path}")
        output_file = processor.save_summaries(
            summaries,
            images_by_chunk,
            job.output_path,
            doc_name,
            doc_type,
            doc_metadata,
            obsidian_metadata
        )

        # Handle image directory
        if settings.get('extract_images', True):
            images_dir = os.path.join(os.path.dirname(job.file_path), f"{doc_name}_images")
            if os.path.exists(images_dir):
                result_images_dir = os.path.join(results_dir, f"{job.job_id}_images")
                shutil.copytree(images_dir, result_images_dir)
                job.add_log(f"Copied images to {result_images_dir}")

                # Add images directory to result files
                job.result_files.append({
                    "type": "directory",
                    "name": f"{doc_name}_images",
                    "path": f"{job.job_id}_images"
                })

        # Add summary file to result files
        job.result_files.append({
            "type": "file",
            "name": os.path.basename(job.output_path),
            "path": os.path.basename(job.output_path)
        })

        # Copy summary file to results directory
        shutil.copy2(job.output_path, os.path.join(results_dir, os.path.basename(job.output_path)))

        # Save to Obsidian if requested
        if settings.get('use_obsidian', False) and settings.get('obsidian_vault_path'):
            obsidian_path = settings.get('obsidian_vault_path')
            if os.path.exists(obsidian_path):
                job.add_log(f"Saving to Obsidian vault: {obsidian_path}")
                obsidian_file = save_to_obsidian(job.output_path, obsidian_path, job, settings)
                if obsidian_file:
                    job.result_files.append({
                        "type": "obsidian",
                        "name": f"Obsidian: {os.path.basename(obsidian_file)}",
                        "path": obsidian_file
                    })

        # Clean up temporary files
        temp_files = [
            f for f in os.listdir('.')
            if (f.startswith(f"temp_{doc_name}_chunk_") or f.startswith(f"temp_{doc_name}_chapter_"))
               and f.endswith(".md")
        ]
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                job.add_log(f"Removed temporary file: {temp_file}")
            except Exception as e:
                job.add_log(f"Could not remove temporary file {temp_file}: {e}")

        # Report any failed chunks
        if job.failed_chunks:
            job.add_log(f"⚠️ Note: {len(job.failed_chunks)} chunks failed processing.")

            # Save failed chunks for future retry
            failed_chunks_file = os.path.join(results_dir, f"{job.job_id}_failed_chunks.json")
            with open(failed_chunks_file, "w", encoding="utf-8") as f:
                json.dump(job.failed_chunks, f, ensure_ascii=False, indent=2)

            job.add_log(f"Failed chunks saved to {failed_chunks_file}")

            # Add failed chunks file to result files
            job.result_files.append({
                "type": "file",
                "name": f"{doc_name}_failed_chunks.json",
                "path": f"{job.job_id}_failed_chunks.json"
            })

            # Add option to retry failed chunks
            job.result_files.append({
                "type": "action",
                "name": f"Retry {len(job.failed_chunks)} Failed Chunks",
                "action": "retry_chunks",
                "job_id": job.job_id
            })

        job.status = "completed"
        job.progress = 100
        job.add_log("✅ Processing completed successfully!")

    except Exception as e:
        error_message = f"Error processing document: {str(e)}"
        job.status = "failed"
        job.error = error_message
        job.add_log(f"❌ {error_message}")
        logger.error(error_message, exc_info=True)

    # Restore stdout
    sys.stdout = original_stdout


def process_retry_chunks(job_id):
    """Process only the failed chunks from a previous job."""
    job = jobs[job_id]
    original_job_id = job.retry_for_job_id

    if not original_job_id or original_job_id not in jobs:
        job.add_log("❌ Original job not found")
        job.status = "failed"
        job.error = "Original job not found"
        return

    original_job = jobs[original_job_id]

    if not original_job.failed_chunks:
        job.add_log("No failed chunks to retry")
        job.status = "completed"
        return

    # Start processing
    job.status = "processing"
    job.add_log(f"Retrying {len(original_job.failed_chunks)} failed chunks from job {original_job_id}")

    # Initialize progress tracking
    job.total_chunks = len(original_job.failed_chunks)
    job.current_chunk = 0
    job.update_progress()

    # Redirect stdout to capture logs
    original_stdout = sys.stdout
    sys.stdout = capture_output(job)

    try:
        # Load original settings with adjustments for retry
        settings = job.settings.copy()

        # Use more aggressive settings for retries
        settings['chunk_timeout'] = int(settings.get('chunk_timeout', 300)) * 2
        settings['api_timeout'] = int(settings.get('api_timeout', 60)) * 2
        settings['max_retries'] = int(settings.get('max_retries', 3)) + 2

        # Initialize processor
        processor = GeminiDocumentProcessor(
            chunk_size=int(settings.get('chunk_size', 7)),
            api_key=settings.get('api_key', ""),
            model_name=settings.get('model_name', "gemini-2.0-flash"),
            language="thai",  # Always Thai for this tool
            max_retries=int(settings.get('max_retries', 5)),  # More retries
            extract_images=settings.get('extract_images', True),
            min_img_width=int(settings.get('min_img_width', 100)),
            min_img_height=int(settings.get('min_img_height', 100)),
            img_format=settings.get('img_format', "png"),
            max_workers=int(settings.get('max_workers', 4)),
            request_timeout=int(settings.get('api_timeout', 120))  # Longer timeout
        )

        # Load the original summaries file
        original_output_path = original_job.output_path

        # Read original file to get existing summaries
        with open(original_output_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        # Process failed chunks
        successful_retries = 0
        for idx, failed_chunk in enumerate(original_job.failed_chunks):
            chunk_num = failed_chunk.get('chunk_number')
            if not chunk_num:
                continue

            # Update progress
            job.current_chunk = idx + 1
            job.update_progress()

            # Try to extract page range from failed chunk info
            page_start = failed_chunk.get('page_start', 0)
            page_end = failed_chunk.get('page_end', 0)

            # If we don't have page info, try to calculate it
            if page_start <= 0 or page_end <= 0:
                chunk_size = int(settings.get('chunk_size', 7))
                page_start = (chunk_num - 1) * chunk_size + 1
                page_end = chunk_num * chunk_size

            job.add_log(f"Retrying chunk {chunk_num} (pages {page_start}-{page_end})...")

            try:
                # Process this chunk with extended timeout
                summary, images = processor.process_chunk(
                    job.file_path,
                    page_start,
                    page_end,
                    timeout=int(settings.get('api_timeout', 120))
                )

                # Save temporary result
                temp_file = f"temp_retry_{job.job_id}_chunk_{chunk_num}.md"
                with open(temp_file, "w", encoding="utf-8") as f:
                    f.write(summary)

                job.add_log(f"  Successfully reprocessed chunk {chunk_num}")
                successful_retries += 1

                # We would need to update the original content here
                # This is a simplistic approach - in practice, you'd need more sophisticated parsing
                chunk_marker = f"### Chunk {chunk_num}"
                next_chunk_marker = f"### Chunk {chunk_num + 1}"

                if chunk_marker in original_content:
                    start_idx = original_content.find(chunk_marker)
                    end_idx = original_content.find(next_chunk_marker, start_idx)

                    if end_idx == -1:  # Last chunk
                        end_idx = original_content.find("---\n*Summary generated", start_idx)

                    if start_idx != -1 and end_idx != -1:
                        # Replace the chunk content
                        new_content = original_content[:start_idx] + chunk_marker + "\n\n" + summary + "\n\n"
                        if end_idx != -1:
                            new_content += original_content[end_idx:]

                        original_content = new_content

            except Exception as e:
                job.add_log(f"❌ Error retrying chunk {chunk_num}: {str(e)}")
                job.failed_chunks.append({
                    "chunk_number": chunk_num,
                    "page_start": page_start,
                    "page_end": page_end,
                    "error": str(e)
                })

        # Save the updated content to a new file
        new_output_path = job.output_path
        with open(new_output_path, 'w', encoding='utf-8') as f:
            f.write(original_content)

        # Copy to results directory
        shutil.copy2(new_output_path, os.path.join(results_dir, os.path.basename(new_output_path)))

        # Add to result files
        job.result_files.append({
            "type": "file",
            "name": os.path.basename(new_output_path),
            "path": os.path.basename(new_output_path)
        })

        # Summary of retry results
        if successful_retries > 0:
            job.add_log(f"✅ Successfully reprocessed {successful_retries} of {len(original_job.failed_chunks)} chunks")
        else:
            job.add_log("❌ Failed to reprocess any chunks")

        # Set final progress and status
        job.status = "completed"
        job.progress = 100

    except Exception as e:
        error_message = f"Error during retry processing: {str(e)}"
        job.status = "failed"
        job.error = error_message
        job.add_log(f"❌ {error_message}")
        logger.error(error_message, exc_info=True)

    # Restore stdout
    sys.stdout = original_stdout

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', obsidian_dir=obsidian_dir)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and job creation"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Check valid file types
    if not (file.filename.lower().endswith('.pdf') or file.filename.lower().endswith('.epub')):
        return jsonify({'error': 'File must be a PDF or EPUB'}), 400

    # Generate a unique job ID
    job_id = str(uuid.uuid4())

    # Save the uploaded file
    file_filename = os.path.basename(file.filename)
    file_name_without_ext = os.path.splitext(file_filename)[0]
    file_path = os.path.join(uploads_dir, f"{job_id}_{file_filename}")
    file.save(file_path)

    # Get settings from form
    settings = {
        'model_name': request.form.get('model_name', 'gemini-2.0-flash'),
        'chunk_size': request.form.get('chunk_size', '7'),
        'api_key': request.form.get('api_key', ''),
        'extract_images': request.form.get('extract_images') == 'on',
        'max_retries': request.form.get('max_retries', '3'),
        'min_img_width': request.form.get('min_img_width', '100'),
        'min_img_height': request.form.get('min_img_height', '100'),
        'img_format': request.form.get('img_format', 'png'),
        'max_workers': request.form.get('max_workers', '4'),
        'chunk_timeout': request.form.get('chunk_timeout', '300'),  # 5 minutes default
        'api_timeout': request.form.get('api_timeout', '60'),  # 1 minute default

        # Obsidian settings
        'use_obsidian': request.form.get('use_obsidian') == 'on',
        'obsidian_vault_path': request.form.get('obsidian_vault_path', ''),
        'obsidian_tags': request.form.get('obsidian_tags', 'book,main,verified'),
        'obsidian_author': request.form.get('obsidian_author', ''),
        'obsidian_cover_url': request.form.get('obsidian_cover_url', ''),
        'obsidian_review': request.form.get('obsidian_review', '')
    }

    # Save Obsidian path to settings if provided
    if settings['obsidian_vault_path'] and os.path.exists(settings['obsidian_vault_path']):
        try:
            settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.json')
            settings_data = {}
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    settings_data = json.load(f)

            settings_data['obsidian_vault_path'] = settings['obsidian_vault_path']

            with open(settings_file, 'w') as f:
                json.dump(settings_data, f, indent=2)

            global obsidian_dir
            obsidian_dir = settings['obsidian_vault_path']
            logger.info(f"Saved Obsidian vault path to settings: {obsidian_dir}")
        except Exception as e:
            logger.error(f"Error saving settings: {e}")

    # Create output path
    output_path = os.path.join(uploads_dir, f"{job_id}_{file_name_without_ext}_summary.md")

    # Create and store the job
    job = ProcessingJob(job_id, file_path, output_path, settings)
    jobs[job_id] = job

    # Start processing in a background thread
    job.thread = threading.Thread(target=process_document_job, args=(job_id,))
    job.thread.daemon = True
    job.thread.start()

    return jsonify({'job_id': job_id, 'redirect': url_for('job_status', job_id=job_id)})


@app.route('/job/<job_id>')
def job_status(job_id):
    """Show job status page"""
    if job_id not in jobs:
        return "Job not found", 404

    return render_template('job_status.html', job_id=job_id)


@app.route('/api/job/<job_id>')
def api_job_status(job_id):
    """API to get job status"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    return jsonify(jobs[job_id].to_dict())


@app.route('/download/<path:filename>')
def download_file(filename):
    """Serve result files for download"""
    return send_from_directory(results_dir, filename, as_attachment=True)


@app.route('/view/<path:filename>')
def view_file(filename):
    """View result files in browser"""
    return send_from_directory(results_dir, filename)


@app.route('/retry_chunks/<job_id>', methods=['POST'])
def retry_failed_chunks(job_id):
    """Retry processing failed chunks for a job."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]

    if not hasattr(job, 'failed_chunks') or not job.failed_chunks:
        return jsonify({'message': 'No failed chunks to retry'}), 200

    # Create a new job for retry
    retry_job_id = str(uuid.uuid4())
    file_name_without_ext = os.path.splitext(os.path.basename(job.file_path))[0]
    retry_output_path = os.path.join(uploads_dir, f"{retry_job_id}_{file_name_without_ext}_summary_retry.md")

    retry_job = ProcessingJob(
        retry_job_id,
        job.file_path,
        retry_output_path,
        job.settings
    )

    # Set special flag to indicate this is a retry job
    retry_job.retry_for_job_id = job_id
    retry_job.failed_chunks = job.failed_chunks.copy()

    # Store the job
    jobs[retry_job_id] = retry_job

    # Start processing in a background thread
    retry_job.thread = threading.Thread(target=process_retry_chunks, args=(retry_job_id,))
    retry_job.thread.daemon = True
    retry_job.thread.start()

    return jsonify({'job_id': retry_job_id, 'redirect': url_for('job_status', job_id=retry_job_id)})


@app.route('/obsidian_check', methods=['POST'])
def check_obsidian_path():
    """Check if the Obsidian vault path is valid"""
    path = request.json.get('path', '')
    if not path:
        return jsonify({'valid': False, 'message': 'Path is empty'})

    if not os.path.exists(path):
        return jsonify({'valid': False, 'message': 'Path does not exist'})

    if not os.path.isdir(path):
        return jsonify({'valid': False, 'message': 'Path is not a directory'})

    # Check for .obsidian folder to confirm it's an Obsidian vault
    obsidian_folder = os.path.join(path, '.obsidian')
    if not os.path.exists(obsidian_folder) or not os.path.isdir(obsidian_folder):
        return jsonify({'valid': False, 'message': 'Not an Obsidian vault (no .obsidian folder)'})

    return jsonify({'valid': True, 'message': 'Valid Obsidian vault'})


if __name__ == '__main__':
    # Create HTML templates folder and files if they don't exist
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    os.makedirs(templates_dir, exist_ok=True)

    # Create index.html template with additional timeout settings
    index_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Gemini Document Processor</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { padding-top: 20px; }
            .card { margin-bottom: 20px; }
            .form-group { margin-bottom: 15px; }
            .obsidian-settings { 
                background-color: #f5f5f5;
                border-radius: 8px;
                padding: 15px;
                margin-top: 15px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="mb-4">Gemini Document Processor</h1>

            <div class="card">
                <div class="card-header">
                    <ul class="nav nav-tabs card-header-tabs" id="myTab" role="tablist">
                        <li class="nav-item" role="presentation">
                            <button class="nav-link active" id="basic-tab" data-bs-toggle="tab" data-bs-target="#basic" type="button" role="tab" aria-controls="basic" aria-selected="true">Basic Settings</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="obsidian-tab" data-bs-toggle="tab" data-bs-target="#obsidian" type="button" role="tab" aria-controls="obsidian" aria-selected="false">Obsidian Integration</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="advanced-tab" data-bs-toggle="tab" data-bs-target="#advanced" type="button" role="tab" aria-controls="advanced" aria-selected="false">Advanced Settings</button>
                        </li>
                    </ul>
                </div>
                <div class="card-body">
                    <form id="uploadForm" enctype="multipart/form-data">
                        <div class="tab-content" id="myTabContent">
                            <div class="tab-pane fade show active" id="basic" role="tabpanel" aria-labelledby="basic-tab">
                                <div class="form-group">
                                    <label for="file">Select PDF or EPUB File:</label>
                                    <input type="file" class="form-control" id="file" name="file" accept=".pdf,.epub" required>
                                    <small class="form-text text-muted">You can upload either a PDF or EPUB file for processing</small>
                                </div>

                                <div class="form-group">
                                    <label for="model_name">Gemini Model:</label>
                                    <select class="form-select" id="model_name" name="model_name">
                                        <option value="gemini-2.0-flash">gemini-2.0-flash (Faster)</option>
                                        <option value="gemini-2.5-flash-preview-04-17">gemini-2.5-flash-preview-04-17 (More accurate)</option>
                                        <option value="gemini-1.5-pro">gemini-1.5-pro (Backup option)</option>
                                    </select>
                                </div>

                                <div class="form-group">
                                    <label for="chunk_size">Pages per Chunk (PDF) / Content per Chapter (EPUB):</label>
                                    <input type="number" class="form-control" id="chunk_size" name="chunk_size" value="7" min="1" max="20">
                                    <small class="form-text text-muted">For PDFs: Number of pages to process in each API call. For EPUBs: Controls how much content is processed at once.</small>
                                </div>

                                <div class="form-group">
                                    <label for="api_key">API Key:</label>
                                    <input type="password" class="form-control" id="api_key" name="api_key" value="">
                                    <div class="form-check mt-2">
                                        <input class="form-check-input" type="checkbox" id="showApiKey">
                                        <label class="form-check-label" for="showApiKey">Show API Key</label>
                                    </div>
                                </div>

                                <div class="form-check mb-3">
                                    <input class="form-check-input" type="checkbox" id="extract_images" name="extract_images" checked>
                                    <label class="form-check-label" for="extract_images">Extract Images</label>
                                </div>
                            </div>

                            <div class="tab-pane fade" id="obsidian" role="tabpanel" aria-labelledby="obsidian-tab">
                                <h5 class="mb-3">Obsidian Integration</h5>

                                <div class="form-check mb-3">
                                    <input class="form-check-input" type="checkbox" id="use_obsidian" name="use_obsidian">
                                    <label class="form-check-label" for="use_obsidian">Enable Obsidian Integration</label>
                                </div>

                                <div id="obsidianSettings" class="obsidian-settings d-none">
                                    <div class="form-group">
                                        <label for="obsidian_vault_path">Obsidian Vault Path:</label>
                                        <div class="input-group">
                                            <input type="text" class="form-control" id="obsidian_vault_path" name="obsidian_vault_path" value="{{ obsidian_dir or '' }}">
                                            <button class="btn btn-outline-secondary" type="button" id="checkObsidianPath">Check</button>
                                        </div>
                                        <div id="obsidianPathFeedback" class="form-text"></div>
                                        <small class="form-text text-muted">Path to your Obsidian vault. The summarized document will be saved to a 'books' folder in this location.</small>
                                    </div>

                                    <div class="form-group mt-3">
                                        <label for="obsidian_tags">Tags (comma separated):</label>
                                        <input type="text" class="form-control" id="obsidian_tags" name="obsidian_tags" value="book,main">
                                    </div>

                                    <div class="form-group mt-3">
                                        <label for="obsidian_author">Author (optional):</label>
                                        <input type="text" class="form-control" id="obsidian_author" name="obsidian_author" value="TBD">
                                        <small class="form-text text-muted">If not specified, will use author from document metadata if available</small>
                                    </div>

                                    <div class="form-group mt-3">
                                        <label for="obsidian_cover_url">Cover URL (optional):</label>
                                        <input type="text" class="form-control" id="obsidian_cover_url" name="obsidian_cover_url" value="TBD">
                                    </div>

                                    <div class="form-group mt-3">
                                        <label for="obsidian_review">Review (optional):</label>
                                        <input type="text" class="form-control" id="obsidian_review" name="obsidian_review" value="TBD">
                                        <small class="form-text text-muted">Example: "20/33" or "5/5"</small>
                                    </div>
                                </div>
                            </div>

                            <div class="tab-pane fade" id="advanced" role="tabpanel" aria-labelledby="advanced-tab">
                                <h5 class="mb-3">Advanced Options</h5>

                                <div class="form-group">
                                    <label for="max_retries">Max Retries:</label>
                                    <input type="number" class="form-control" id="max_retries" name="max_retries" value="3" min="1" max="10">
                                    <small class="form-text text-muted">Maximum number of retry attempts for API calls</small>
                                </div>

                                <div class="form-group">
                                    <label for="chunk_timeout">Chunk Processing Timeout (seconds):</label>
                                    <input type="number" class="form-control" id="chunk_timeout" name="chunk_timeout" value="300" min="60" max="1800">
                                    <small class="form-text text-muted">Maximum time allowed for processing a single chunk before it's considered failed (5-30 minutes)</small>
                                </div>

                                <div class="form-group">
                                    <label for="api_timeout">API Request Timeout (seconds):</label>
                                    <input type="number" class="form-control" id="api_timeout" name="api_timeout" value="180" min="30" max="300">
                                    <small class="form-text text-muted">Maximum time allowed for a single API call (30-300 seconds)</small>
                                </div>

                                <div class="form-group">
                                    <label for="min_img_width">Minimum Image Width:</label>
                                    <input type="number" class="form-control" id="min_img_width" name="min_img_width" value="100" min="10" max="1000">
                                    <small class="form-text text-muted">Images smaller than this width will be ignored</small>
                                </div>

                                <div class="form-group">
                                    <label for="min_img_height">Minimum Image Height:</label>
                                    <input type="number" class="form-control" id="min_img_height" name="min_img_height" value="100" min="10" max="1000">
                                    <small class="form-text text-muted">Images smaller than this height will be ignored</small>
                                </div>

                                <div class="form-group">
                                    <label for="img_format">Image Format:</label>
                                    <select class="form-select" id="img_format" name="img_format">
                                        <option value="jpg">JPG (Smaller files, some quality loss)</option>
                                        <option value="png">PNG (Lossless, larger files)</option>
                                    </select>
                                </div>

                                <div class="form-group">
                                    <label for="max_workers">Worker Threads:</label>
                                    <input type="number" class="form-control" id="max_workers" name="max_workers" value="4" min="1" max="16">
                                    <small class="form-text text-muted">Number of parallel threads for image extraction</small>
                                </div>
                            </div>
                        </div>

                        <div class="mt-4">
                            <button type="submit" class="btn btn-primary" id="submitBtn">Process Document</button>
                            <div class="spinner-border text-primary d-none" id="spinner" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                        </div>
                    </form>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // Toggle API key visibility
            document.getElementById('showApiKey').addEventListener('change', function() {
                const apiKeyInput = document.getElementById('api_key');
                apiKeyInput.type = this.checked ? 'text' : 'password';
            });

            // Toggle Obsidian settings visibility
            document.getElementById('use_obsidian').addEventListener('change', function() {
                const obsidianSettings = document.getElementById('obsidianSettings');
                obsidianSettings.classList.toggle('d-none', !this.checked);
            });

            // Check Obsidian path
            document.getElementById('checkObsidianPath').addEventListener('click', function() {
                const path = document.getElementById('obsidian_vault_path').value;
                const feedbackElement = document.getElementById('obsidianPathFeedback');

                feedbackElement.textContent = 'Checking...';
                feedbackElement.className = 'form-text text-muted';

                fetch('/obsidian_check', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ path: path })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.valid) {
                        feedbackElement.textContent = '✅ ' + data.message;
                        feedbackElement.className = 'form-text text-success';
                    } else {
                        feedbackElement.textContent = '❌ ' + data.message;
                        feedbackElement.className = 'form-text text-danger';
                    }
                })
                .catch(error => {
                    feedbackElement.textContent = '❌ Error checking path';
                    feedbackElement.className = 'form-text text-danger';
                });
            });

            // Handle form submission
            document.getElementById('uploadForm').addEventListener('submit', function(e) {
                e.preventDefault();

                const submitBtn = document.getElementById('submitBtn');
                const spinner = document.getElementById('spinner');

                // Disable button and show spinner
                submitBtn.disabled = true;
                spinner.classList.remove('d-none');

                // Create FormData object
                const formData = new FormData(this);

                // Submit the form via AJAX
                fetch('/upload', {
                    method: 'POST',
                    body: formData
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.redirect) {
                        window.location.href = data.redirect;
                    }
                })
                .catch(error => {
                    alert('Error: ' + error.message);
                    submitBtn.disabled = false;
                    spinner.classList.add('d-none');
                });
            });
        </script>
    </body>
    </html>
    """

    # Create job_status.html template with retry buttons
    job_status_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Job Status - Gemini Document Processor</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { padding-top: 20px; }
            .log-container {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 0.25rem;
                padding: 15px;
                height: 300px;
                overflow-y: auto;
                font-family: monospace;
                white-space: pre-wrap;
            }
            .log-line {
                margin-bottom: 3px;
            }
            .badge-obsidian {
                background-color: #8d6fdb;
                color: white;
            }
            .current-chunk-info {
                font-size: 0.9rem;
                color: #6c757d;
                margin-top: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="mb-4">Document Processing Status</h1>

            <div class="card">
                <div class="card-header">
                    <div class="d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">Job Status</h5>
                        <a href="/" class="btn btn-sm btn-outline-secondary">Back to Home</a>
                    </div>
                </div>
                <div class="card-body">
                    <div class="mb-3">
                        <div class="d-flex justify-content-between">
                            <strong>Status:</strong>
                            <span id="status" class="badge bg-secondary">Loading...</span>
                        </div>
                    </div>

                    <div class="mb-3">
                        <strong>Progress:</strong>
                        <div class="progress">
                            <div id="progress" class="progress-bar" role="progressbar" style="width: 0%;" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">0%</div>
                        </div>
                        <div id="chunkInfo" class="current-chunk-info d-none">
                            Processing chunk <span id="currentChunk">0</span> of <span id="totalChunks">0</span>
                        </div>
                    </div>

                    <div class="mb-3">
                        <strong>Current Task:</strong>
                        <div id="message">Initializing...</div>
                    </div>

                    <div class="mb-3">
                        <strong>Log:</strong>
                        <div id="log" class="log-container"></div>
                    </div>

                    <div id="failedChunksContainer" class="mb-3 d-none">
                        <div class="alert alert-warning" role="alert">
                            <strong id="failedChunksCount">0</strong> chunks failed processing. 
                            <button id="retryChunksBtn" class="btn btn-sm btn-warning ms-2">Retry Failed Chunks</button>
                        </div>
                    </div>

                    <div id="resultsContainer" class="mb-3 d-none">
                        <strong>Results:</strong>
                        <div id="results" class="list-group mt-2"></div>
                    </div>

                    <div id="errorContainer" class="alert alert-danger d-none" role="alert">
                        <strong>Error:</strong>
                        <div id="errorMessage"></div>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // Get job ID from URL
            const jobId = '{{ job_id }}';
            let statusCheckInterval;

            // Update status badge based on job status
            function updateStatusBadge(status) {
                const statusElement = document.getElementById('status');
                statusElement.textContent = status.charAt(0).toUpperCase() + status.slice(1);

                // Update badge color
                statusElement.className = 'badge';
                switch(status) {
                    case 'pending':
                        statusElement.classList.add('bg-secondary');
                        break;
                    case 'processing':
                        statusElement.classList.add('bg-primary');
                        break;
                    case 'completed':
                        statusElement.classList.add('bg-success');
                        break;
                    case 'failed':
                        statusElement.classList.add('bg-danger');
                        break;
                    default:
                        statusElement.classList.add('bg-secondary');
                }
            }

            // Update progress bar
            function updateProgress(progress, currentChunk, totalChunks) {
                const progressBar = document.getElementById('progress');
                progressBar.style.width = `${progress}%`;
                progressBar.textContent = `${progress}%`;
                progressBar.setAttribute('aria-valuenow', progress);

                // Update chunk info if available
                const chunkInfo = document.getElementById('chunkInfo');
                if (currentChunk && totalChunks) {
                    document.getElementById('currentChunk').textContent = currentChunk;
                    document.getElementById('totalChunks').textContent = totalChunks;
                    chunkInfo.classList.remove('d-none');
                } else {
                    chunkInfo.classList.add('d-none');
                }
            }

            // Update log
            function updateLog(logEntries) {
                const logContainer = document.getElementById('log');

                // Clear existing log
                logContainer.innerHTML = '';

                // Add new log entries
                logEntries.forEach(entry => {
                    const logLine = document.createElement('div');
                    logLine.className = 'log-line';

                    // Add error styling if needed
                    if (entry.includes('❌') || entry.includes('Error')) {
                        logLine.className += ' text-danger';
                    } else if (entry.includes('⚠️') || entry.includes('WARNING')) {
                        logLine.className += ' text-warning';
                    } else if (entry.includes('✅')) {
                        logLine.className += ' text-success';
                    }

                    logLine.textContent = entry;
                    logContainer.appendChild(logLine);
                });

                // Auto-scroll to bottom
                logContainer.scrollTop = logContainer.scrollHeight;
            }

            // Update failed chunks info
            function updateFailedChunks(failedChunksCount) {
                const container = document.getElementById('failedChunksContainer');
                const countElement = document.getElementById('failedChunksCount');

                if (failedChunksCount > 0) {
                    countElement.textContent = failedChunksCount;
                    container.classList.remove('d-none');
                } else {
                    container.classList.add('d-none');
                }
            }

            // Update results
            function updateResults(resultFiles) {
                const resultsContainer = document.getElementById('resultsContainer');
                const resultsList = document.getElementById('results');

                if (resultFiles && resultFiles.length > 0) {
                    // Show results container
                    resultsContainer.classList.remove('d-none');

                    // Clear existing results
                    resultsList.innerHTML = '';

                    // Add new result files
                    resultFiles.forEach(file => {
                        const resultItem = document.createElement('a');
                        resultItem.className = 'list-group-item list-group-item-action d-flex justify-content-between align-items-center';

                        if (file.type === 'file') {
                            resultItem.href = `/download/${file.path}`;
                            resultItem.textContent = file.name;

                            // Add view button for markdown files
                            if (file.name.endsWith('.md')) {
                                const viewBtn = document.createElement('a');
                                viewBtn.className = 'btn btn-sm btn-outline-primary ms-2';
                                viewBtn.href = `/view/${file.path}`;
                                viewBtn.textContent = 'View';
                                viewBtn.target = '_blank';
                                resultItem.appendChild(viewBtn);
                            }
                        } else if (file.type === 'directory') {
                            resultItem.textContent = `${file.name} (directory)`;
                            resultItem.href = '#';
                            resultItem.style.pointerEvents = 'none';
                        } else if (file.type === 'obsidian') {
                            resultItem.className += ' list-group-item-light';
                            resultItem.innerHTML = `<span>${file.name}</span>`;
                            resultItem.href = '#';

                            // Add obsidian badge
                            const badge = document.createElement('span');
                            badge.className = 'badge badge-obsidian';
                            badge.textContent = 'Obsidian';
                            resultItem.appendChild(badge);
                        } else if (file.type === 'action' && file.action === 'retry_chunks') {
                            resultItem.href = '#';
                            resultItem.textContent = file.name;
                            resultItem.className = 'list-group-item list-group-item-warning d-flex justify-content-between align-items-center';

                            // Add retry button
                            const retryBtn = document.createElement('button');
                            retryBtn.className = 'btn btn-sm btn-warning';
                            retryBtn.textContent = 'Retry';
                            retryBtn.onclick = function(e) {
                                e.preventDefault();
                                retryFailedChunks(file.job_id);
                            };
                            resultItem.appendChild(retryBtn);
                        }

                        resultsList.appendChild(resultItem);
                    });
                }
            }

            // Show error message
            function showError(errorMessage) {
                const errorContainer = document.getElementById('errorContainer');
                const errorMessageElement = document.getElementById('errorMessage');

                errorContainer.classList.remove('d-none');
                errorMessageElement.textContent = errorMessage;
            }

            // Function to retry failed chunks
            function retryFailedChunks(originalJobId) {
                if (!confirm('Are you sure you want to retry failed chunks?')) {
                    return;
                }

                fetch(`/retry_chunks/${originalJobId}`, {
                    method: 'POST'
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Failed to start retry job');
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.redirect) {
                        window.location.href = data.redirect;
                    }
                })
                .catch(error => {
                    alert('Error starting retry job: ' + error.message);
                });
            }

            // Check job status
            function checkJobStatus() {
                fetch(`/api/job/${jobId}`)
                    .then(response => {
                        if (!response.ok) {
                            throw new Error('Failed to get job status');
                        }
                        return response.json();
                    })
                    .then(data => {
                        // Update UI with job status
                        updateStatusBadge(data.status);
                        updateProgress(data.progress, data.current_chunk, data.total_chunks);
                        document.getElementById('message').textContent = data.message;
                        updateLog(data.log);
                        updateFailedChunks(data.failed_chunks);

                        // If job is completed, show results
                        if (data.result_files && data.result_files.length > 0) {
                            updateResults(data.result_files);
                        }

                        // If job has an error, show it
                        if (data.error) {
                            showError(data.error);
                        }

                        // If job is completed or failed, stop checking
                        if (data.status === 'completed' || data.status === 'failed') {
                            clearInterval(statusCheckInterval);
                            updateStatusBadge(data.status);
                        }
                    })
                    .catch(error => {
                        console.error('Error checking job status:', error);
                        document.getElementById('message').textContent = 'Error checking job status';
                    });
            }

            // Start checking job status
            document.addEventListener('DOMContentLoaded', function() {
                // Initial check
                checkJobStatus();

                // Check status every 5 seconds
                statusCheckInterval = setInterval(checkJobStatus, 5000);

                // Set up retry button
                document.getElementById('retryChunksBtn').addEventListener('click', function() {
                    retryFailedChunks(jobId);
                });
            });
        </script>
    </body>
    </html>
    """

    with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
        f.write(index_html)

    with open(os.path.join(templates_dir, 'job_status.html'), 'w') as f:
        f.write(job_status_html)

    print("Starting web server at http://127.0.0.1:8081/")
    print("You can access the document processor by opening this URL in your browser")
    app.run(debug=True, port=8081)