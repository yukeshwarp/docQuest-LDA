import fitz
import io
import base64
import logging
import string
import nltk
from celery import Celery
from concurrent.futures import ThreadPoolExecutor, as_completed
from nltk.corpus import stopwords
from utils.file_conversion import convert_office_to_pdf
from utils.llm_interaction import (
    summarize_page,
    get_image_explanation,
    generate_system_prompt,
)
from utils.config import redis_host, redis_pass

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s [%(levelname)s] %(message)s"
)

redis_host = redis_host
redis_port = 6379
redis_password = redis_pass

app = Celery(
    "pdf_processor",
    broker=f"redis://:{redis_password}@{redis_host}:{redis_port}/0",
    backend=f"redis://:{redis_password}@{redis_host}:{redis_port}/0",
)

app.conf.update(
    result_expires=3600,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)

generated_system_prompt = None


def remove_stopwords_and_blanks(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    filtered_text = " ".join(
        [word for word in text.split() if word.lower() not in stop_words]
    )
    return " ".join(filtered_text.split())


def detect_ocr_images_and_vector_graphics_in_pdf(page, ocr_text_threshold=0.4):
    try:
        images = page.get_images(full=True)
        text_blocks = page.get_text("blocks")
        vector_graphics_detected = bool(page.get_drawings())

        page_area = page.rect.width * page.rect.height
        text_area = sum(
            (block[2] - block[0]) * (block[3] - block[1]) for block in text_blocks
        )
        text_coverage = text_area / page_area if page_area > 0 else 0

        pix = page.get_pixmap()
        img_data = pix.tobytes("png")
        base64_image = base64.b64encode(img_data).decode("utf-8")
        pix = None

        if (images or vector_graphics_detected) and text_coverage < ocr_text_threshold:
            return base64_image

    except Exception as e:
        logging.error(f"Error detecting OCR images/graphics on page {page.number}: {e}")

    return None


def process_page_batch(pdf_document, batch, system_prompt, ocr_text_threshold=0.4):
    previous_summary = ""
    batch_data = []

    for page_number in batch:
        try:
            page = pdf_document.load_page(page_number)
            text = page.get_text("text").strip()
            summary = ""

            if text != "":
                summary = summarize_page(
                    text, previous_summary, page_number + 1, system_prompt
                )
                previous_summary = summary

            image_data = detect_ocr_images_and_vector_graphics_in_pdf(
                page, ocr_text_threshold
            )
            image_analysis = []
            if image_data:
                image_explanation = get_image_explanation(image_data)
                image_analysis.append(
                    {"page_number": page_number + 1, "explanation": image_explanation}
                )

            batch_data.append(
                {
                    "page_number": page_number + 1,
                    "full_text": text,
                    "text_summary": summary,
                    "image_analysis": image_analysis,
                }
            )

        except Exception as e:
            logging.error(f"Error processing page {page_number + 1}: {e}")
            batch_data.append(
                {
                    "page_number": page_number + 1,
                    "full_text": "",
                    "text_summary": "Error in processing this page",
                    "image_analysis": [],
                }
            )

    return batch_data


def process_pdf_pages(uploaded_file, first_file=False):
    global generated_system_prompt
    file_name = uploaded_file.name

    try:
        if file_name.lower().endswith(".pdf"):
            pdf_stream = io.BytesIO(uploaded_file.read())
        else:
            pdf_stream = convert_office_to_pdf(uploaded_file)

        pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
        document_data = {"document_name": file_name, "pages": []}
        total_pages = len(pdf_document)
        full_text = ""

        if first_file and generated_system_prompt is None:
            for page_number in range(total_pages):
                page = pdf_document.load_page(page_number)
                full_text += page.get_text("text").strip() + " "
                if len(full_text.split()) >= 200:
                    break

            first_200_words = " ".join(full_text.split()[:200])
            generated_system_prompt = generate_system_prompt(first_200_words)

        batch_size = 5
        page_batches = [
            range(i, min(i + batch_size, total_pages))
            for i in range(0, total_pages, batch_size)
        ]

        with ThreadPoolExecutor() as executor:
            future_to_batch = {
                executor.submit(
                    process_page_batch, pdf_document, batch, generated_system_prompt
                ): batch
                for batch in page_batches
            }
            for future in as_completed(future_to_batch):
                try:
                    batch_data = future.result()
                    document_data["pages"].extend(batch_data)
                except Exception as e:
                    logging.error(f"Error processing batch: {e}")

        pdf_document.close()
        document_data["pages"].sort(key=lambda x: x["page_number"])
        return document_data

    except Exception as e:
        logging.error(f"Error processing PDF file {file_name}: {e}")
        raise ValueError(f"Unable to process the file {file_name}. Error: {e}")


@app.task
def process_pdf_task(uploaded_file, first_file=False):
    try:
        result = process_pdf_pages(uploaded_file, first_file)
        return result
    except Exception as e:
        logging.error(f"Failed to process PDF: {e}")
        raise e
