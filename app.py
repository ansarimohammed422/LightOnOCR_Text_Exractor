# import gradio as gr
# import spaces
# from gradio.themes.base import Base
# from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info
# from PIL import Image
# from datetime import datetime
# import os
# import json
# import fitz  # PyMuPDF


# class CustomTheme(Base):
#     def __init__(self):
#         super().__init__()
#         self.primary_hue = "blue"
#         self.secondary_hue = "sky"


# custom_theme = CustomTheme()

# DESCRIPTION = "A powerful vision-language model that can understand images and text to provide detailed analysis."


# def array_to_image_path(image_filepath, max_width=1250, max_height=1750):
#     if image_filepath is None:
#         raise ValueError("No image provided.")

#     img = Image.open(image_filepath)
#     width, height = img.size
#     if width > max_width or height > max_height:
#         img.thumbnail((max_width, max_height))

#     return os.path.abspath(image_filepath), img.width, img.height


# def convert_pdf_to_images(pdf_path):
#     """Opens a PDF and converts each page into a high-resolution PNG image."""
#     image_paths = []
#     doc = fitz.open(pdf_path)
#     base_name = os.path.splitext(os.path.basename(pdf_path))[0]

#     for i, page in enumerate(doc):
#         pix = page.get_pixmap(dpi=200)
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         image_path = f"{base_name}_page_{i + 1}_{timestamp}.png"
#         pix.save(image_path)
#         image_paths.append(image_path)

#     doc.close()
#     return image_paths


# # Initialize the model and processor
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
# )
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")


# @spaces.GPU
# def run_inference(uploaded_files, text_input):
#     results = []
#     temp_files_to_clean = []

#     json_prompt = (
#         f"{text_input}\n\nBased on the image and the query, respond ONLY with a single, "
#         "valid JSON object. This object should be well-structured, using nested objects "
#         "and arrays to logically represent the information."
#     )

#     if not uploaded_files:
#         error_json = json.dumps(
#             {"error": "No file provided. Please upload an image or PDF."}, indent=4
#         )
#         return error_json, gr.Button(interactive=False)

#     image_paths_to_process = []
#     unsupported_files = []
#     for file_obj in uploaded_files:
#         file_path = file_obj.name
#         temp_files_to_clean.append(file_path)

#         if file_path.lower().endswith(".pdf"):
#             pdf_page_images = convert_pdf_to_images(file_path)
#             image_paths_to_process.extend(pdf_page_images)
#             temp_files_to_clean.extend(pdf_page_images)
#         elif file_path.lower().endswith(
#             (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
#         ):
#             image_paths_to_process.append(file_path)
#         else:
#             unsupported_files.append(os.path.basename(file_path))

#     if unsupported_files:
#         unsupported_str = ", ".join(unsupported_files)
#         results.append(
#             json.dumps(
#                 {
#                     "error": f"Unsupported file type(s) were ignored: {unsupported_str}",
#                     "details": "Please upload only images (PNG, JPG, etc.) or PDF files.",
#                 },
#                 indent=4,
#             )
#         )

#     for image_file in image_paths_to_process:
#         try:
#             image_path, width, height = array_to_image_path(image_file)

#             messages = [
#                 {
#                     "role": "user",
#                     "content": [
#                         {
#                             "type": "image",
#                             "image": image_path,
#                             "resized_height": height,
#                             "resized_width": width,
#                         },
#                         {"type": "text", "text": json_prompt},
#                     ],
#                 }
#             ]
#             text = processor.apply_chat_template(
#                 messages, tokenize=False, add_generation_prompt=True
#             )
#             image_inputs, video_inputs = process_vision_info(messages)
#             inputs = processor(
#                 text=[text],
#                 images=image_inputs,
#                 videos=video_inputs,
#                 padding=True,
#                 return_tensors="pt",
#             ).to("cuda")

#             generated_ids = model.generate(**inputs, max_new_tokens=4096)
#             generated_ids_trimmed = [
#                 out_ids[len(in_ids) :]
#                 for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#             ]
#             raw_output = processor.batch_decode(
#                 generated_ids_trimmed,
#                 skip_special_tokens=True,
#                 clean_up_tokenization_spaces=True,
#             )
#             raw_text = raw_output[0]

#             try:
#                 start_index = raw_text.find("{")
#                 end_index = raw_text.rfind("}") + 1
#                 if start_index != -1 and end_index != 0:
#                     json_string = raw_text[start_index:end_index]
#                     parsed_json = json.loads(json_string)
#                     parsed_json["source_page"] = os.path.basename(image_path)
#                     formatted_json = json.dumps(parsed_json, indent=4)
#                     results.append(formatted_json)
#                 else:
#                     results.append(
#                         f'{{"error": "Model did not return valid JSON.", "source_page": "{os.path.basename(image_path)}", "raw_response": "{raw_text}"}}'
#                     )
#             except json.JSONDecodeError:
#                 results.append(
#                     f'{{"error": "Failed to decode JSON.", "source_page": "{os.path.basename(image_path)}", "raw_response": "{raw_text}"}}'
#                 )
#         except Exception as e:
#             results.append(
#                 f'{{"error": "An unexpected error occurred during processing.", "details": "{str(e)}"}}'
#             )

#     for f in temp_files_to_clean:
#         if os.path.exists(f):
#             try:
#                 os.remove(f)
#             except OSError as e:
#                 print(f"Error deleting file {f}: {e}")

#     final_json = "\n---\n".join(results)
#     is_error = '"error":' in final_json
#     return final_json, gr.Button(interactive=not is_error)


# @spaces.GPU
# def generate_explanation(json_text):
#     if not json_text or '"error":' in json_text:
#         return "Cannot generate an explanation. Please produce a valid JSON output first. 🙁"

#     explanation_prompt = (
#         "You are an expert data analyst. Your task is to provide a comprehensive, human-readable explanation "
#         "of the following JSON data, which may represent one or more pages from a document. First, provide a textual explanation. "
#         "If the JSON contains data from multiple sources (pages), explain each one. Then, if the JSON data represents a table, "
#         "a list of items, or a receipt, you **must** re-format the key information into a Markdown table for clarity.\n\n"
#         f"JSON Data:\n```json\n{json_text}\n```"
#     )

#     messages = [{"role": "user", "content": explanation_prompt}]
#     text = processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     inputs = processor(text=[text], return_tensors="pt").to("cuda")

#     generated_ids = model.generate(**inputs, max_new_tokens=2048)
#     generated_ids_trimmed = [
#         out_ids[len(in_ids) :]
#         for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#     ]
#     explanation_output = processor.batch_decode(
#         generated_ids_trimmed,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=True,
#     )[0]

#     return explanation_output


# # Define the Gradio UI
# css = """
#   .gradio-container { font-family: 'IBM Plex Sans', sans-serif; }

#   /* Default (Light Mode) Styles */
#   #output-code, #output-code pre, #output-code code {
#     background-color: #f0f0f0;
#     border: 1px solid #e0e0e0;
#     border-radius: 7px;
#     color: #333;
#   }
#   #output-code .token.punctuation { color: #393a34; }
#   #output-code .token.property, #output-code .token.string { color: #0b7500; }
#   #output-code .token.number { color: #2973b7; }
#   #output-code .token.boolean { color: #9a050f; }

#   #explanation-box {
#     min-height: 200px;
#     border: 1px solid #e0e0e0;
#     padding: 15px;
#     border-radius: 7px;
#   }

#   /* Dark Mode Overrides targeting Gradio's .dark class */
#   .dark #output-code, .dark #output-code pre, .dark #output-code code {
#     background-color: #2b2b2b !important;
#     border: 1px solid #444 !important;
#     color: #f0f0f0 !important;
#   }
#   .dark #explanation-box {
#     border: 1px solid #444 !important;
#   }
#   .dark #output-code code span {
#      color: #f0f0f0 !important;
#   }
#   .dark #output-code .token.punctuation { color: #ccc !important; }
#   .dark #output-code .token.property, .dark #output-code .token.string { color: #90ee90 !important; }
#   .dark #output-code .token.number { color: #add8e6 !important; }
#   .dark #output-code .token.boolean { color: #f08080 !important; }
# """

# with gr.Blocks(theme=custom_theme, css=css) as demo:
#     gr.Markdown("# Sparrow Qwen2-VL-7B Vision AI 👁️")
#     gr.Markdown(DESCRIPTION)

#     with gr.Row():
#         with gr.Column(scale=1):
#             input_files = gr.Files(label="Upload Images or PDFs")
#             text_input = gr.Textbox(
#                 label="Your Query",
#                 placeholder="e.g., Extract the total amount from this receipt.",
#             )
#             submit_btn = gr.Button("Analyze File(s)", variant="primary")

#         with gr.Column(scale=2):
#             output_text = gr.Code(
#                 label="Full JSON Response",
#                 language="json",
#                 elem_id="output-code",
#                 interactive=False,
#             )
#             explanation_btn = gr.Button(
#                 "📄 Generate Detailed Explanation", interactive=False
#             )
#             explanation_output = gr.Markdown(
#                 label="Detailed Explanation", elem_id="explanation-box"
#             )

#     # Add api_name to create stable API endpoints
#     submit_btn.click(
#         fn=run_inference,
#         inputs=[input_files, text_input],
#         outputs=[output_text, explanation_btn],
#         api_name="analyze_document",
#     )

#     explanation_btn.click(
#         fn=generate_explanation,
#         inputs=[output_text],
#         outputs=[explanation_output],
#         show_progress="full",
#         api_name="generate_explanation",
#     )

# demo.queue()
# demo.launch(debug=True)

# import gradio as gr
# import spaces
# from gradio.themes.base import Base
# from PIL import Image
# from datetime import datetime
# import os
# import fitz  # PyMuPDF
# import base64
# import io
# from openai import OpenAI

# # --- 1. Client for vLLM Server ---
# # Assumes your LightOnOCR vLLM server is running at this address
# try:
#     client = OpenAI(
#         base_url="http://127.0.0.1:8000/v1",
#         api_key="vllm",  # API key can be anything
#     )
#     # Test connection
#     client.models.list()
#     print("Successfully connected to vLLM server at http://127.0.0.1:8000")
# except Exception as e:
#     print(f"---" * 20)
#     print(f"⚠️ ERROR: Could not connect to vLLM server at http://127.0.0.1:8000")
#     print(f"Please make sure the vLLM server is running in a separate terminal.")
#     print(f"Error details: {e}")
#     print(f"---" * 20)
#     # Gradio will still launch, but the button will show an error on click.


# # --- 2. Theme & Description ---
# class CustomTheme(Base):
#     def __init__(self):
#         super().__init__()
#         self.primary_hue = "blue"
#         self.secondary_hue = "sky"


# custom_theme = CustomTheme()

# DESCRIPTION = """
# This app uses the **LightOnOCR** model (served via vLLM) to perform high-accuracy text extraction.
# <br>
# Upload one or more images or PDFs. The model will read all the text from every page.
# <br>
# **Note:** The 'Your Query' box is not used. This model performs full-text extraction, it does not answer specific questions.
# """

# # --- 3. Utility Functions (Kept from your code) ---


# def convert_pdf_to_images(pdf_path):
#     """Opens a PDF and converts each page into a high-resolution PNG image."""
#     image_paths = []
#     doc = fitz.open(pdf_path)
#     base_name = os.path.splitext(os.path.basename(pdf_path))[0]

#     print(f"Converting PDF: {base_name} ({len(doc)} pages)")
#     for i, page in enumerate(doc):
#         pix = page.get_pixmap(dpi=200)  # High-resolution
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         image_path = f"{base_name}_page_{i + 1}_{timestamp}.png"
#         pix.save(image_path)
#         image_paths.append(image_path)

#     doc.close()
#     return image_paths


# def encode_pil_to_base64(pil_image):
#     """Converts a PIL Image object to a base64 string."""
#     buffered = io.BytesIO()
#     pil_image.save(buffered, format="JPEG")
#     base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
#     return f"data:image/jpeg;base64,{base64_image}"


# # --- 4. Main Inference Function (Adapted for LightOnOCR) ---


# @spaces.GPU
# def run_ocr(uploaded_files, text_input):
#     """
#     Takes uploaded files, processes them, and sends them to the LightOnOCR
#     vLLM server for text extraction.

#     The 'text_input' is ignored as LightOnOCR only performs full-text extraction.
#     """
#     results = []
#     temp_files_to_clean = []

#     if not uploaded_files:
#         return "❌ **Error:** No file provided. Please upload an image or PDF."

#     image_paths_to_process = []
#     unsupported_files = []

#     for file_obj in uploaded_files:
#         file_path = file_obj.name
#         temp_files_to_clean.append(file_path)

#         if file_path.lower().endswith(".pdf"):
#             try:
#                 pdf_page_images = convert_pdf_to_images(file_path)
#                 image_paths_to_process.extend(pdf_page_images)
#                 temp_files_to_clean.extend(pdf_page_images)
#             except Exception as e:
#                 unsupported_files.append(f"{os.path.basename(file_path)} (Error: {e})")

#         elif file_path.lower().endswith(
#             (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
#         ):
#             image_paths_to_process.append(file_path)
#         else:
#             unsupported_files.append(os.path.basename(file_path))

#     if unsupported_files:
#         unsupported_str = ", ".join(unsupported_files)
#         results.append(
#             f"⚠️ **Warning:** Unsupported file type(s) were ignored: {unsupported_str}"
#         )

#     if not image_paths_to_process:
#         return "❌ **Error:** No valid image or PDF files found to process."

#     # --- Process each image with LightOnOCR ---
#     for image_file in image_paths_to_process:
#         try:
#             pil_image = Image.open(image_file)
#             image_url = encode_pil_to_base64(pil_image)

#             # Send to vLLM server
#             response = client.chat.completions.create(
#                 model="lightonai/LightOnOCR-1B-1025",
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": [
#                             {
#                                 "type": "text",
#                                 "text": "Extract all text from this document.",
#                             },
#                             {"type": "image_url", "image_url": {"url": image_url}},
#                         ],
#                     }
#                 ],
#                 max_tokens=4096,
#                 temperature=0.0,
#             )

#             extracted_text = response.choices[0].message.content

#             # Append result with a header for clarity
#             results.append(
#                 f"## 📄 Page: {os.path.basename(image_file)}\n\n"
#                 f"```\n{extracted_text}\n```"
#             )

#         except Exception as e:
#             results.append(
#                 f"## ❌ Error Processing: {os.path.basename(image_file)}\n\n"
#                 f"Details: {str(e)}"
#             )

#     # --- 5. Cleanup (Kept from your code) ---
#     print(f"Cleaning up {len(temp_files_to_clean)} temporary files...")
#     for f in temp_files_to_clean:
#         if os.path.exists(f):
#             try:
#                 os.remove(f)
#             except OSError as e:
#                 print(f"Error deleting file {f}: {e}")

#     # Combine all results into one
#     final_output = "\n\n---\n\n".join(results)
#     return final_output


# # --- 6. Gradio UI (Adapted) ---

# with gr.Blocks(theme=custom_theme) as demo:
#     gr.Markdown("# ⚡ LightOnOCR (via vLLM) Demo ⚡")
#     gr.Markdown(DESCRIPTION)

#     with gr.Row():
#         with gr.Column(scale=1):
#             input_files = gr.Files(label="Upload Images or PDFs")
#             text_input = gr.Textbox(
#                 label="Your Query (Not Used by This Model)",
#                 placeholder="e.g., Extract the total amount from this receipt.",
#             )
#             submit_btn = gr.Button("Extract Text", variant="primary")

#         with gr.Column(scale=2):
#             output_text = gr.Markdown(
#                 label="Extracted Text (All Pages)", elem_id="explanation-box"
#             )

#     submit_btn.click(
#         fn=run_ocr,
#         inputs=[input_files, text_input],
#         outputs=[output_text],
#         api_name="extract_text",
#     )

# demo.queue()
# # demo.launch(debug=True)
# demo.launch(server_name="127.0.0.1", server_port=7860)


import gradio as gr
import spaces
from gradio.themes.base import Base
import uvicorn

# --- Imports for 4-bit Quantization ---
import torch
from transformers import BitsAndBytesConfig
# --- End Quantization Imports ---

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from datetime import datetime
import os
import json
import fitz  # PyMuPDF


class CustomTheme(Base):
    def __init__(self):
        super().__init__()
        self.primary_hue = "blue"
        self.secondary_hue = "sky"


custom_theme = CustomTheme()

DESCRIPTION = "A powerful vision-language model that can understand images and text to provide detailed analysis."


def array_to_image_path(image_filepath, max_width=1250, max_height=1750):
    if image_filepath is None:
        raise ValueError("No image provided.")

    img = Image.open(image_filepath)
    width, height = img.size
    if width > max_width or height > max_height:
        img.thumbnail((max_width, max_height))

    return os.path.abspath(image_filepath), img.width, img.height


def convert_pdf_to_images(pdf_path):
    """Opens a PDF and converts each page into a high-resolution PNG image."""
    image_paths = []
    doc = fitz.open(pdf_path)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=200)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"{base_name}_page_{i + 1}_{timestamp}.png"
        pix.save(image_path)
        image_paths.append(image_path)

    doc.close()
    return image_paths


# --- 1. Define the 4-bit Quantization Configuration ---
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# --- 2. Initialize the Model and Processor with 4-bit Config ---
print("Loading 4-bit quantized model... (This may take a moment)")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    quantization_config=quantization_config,  # <-- This applies the 4-bit config
    device_map="auto",  # <-- This automatically uses the GPU
)
print("Model loaded successfully on GPU.")

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")


@spaces.GPU
def run_inference(uploaded_files, text_input):
    results = []
    temp_files_to_clean = []

    json_prompt = (
        f"{text_input}\n\nBased on the image and the query, respond ONLY with a single, "
        "valid JSON object. This object should be well-structured, using nested objects "
        "and arrays to logically represent the information."
    )

    if not uploaded_files:
        error_json = json.dumps(
            {"error": "No file provided. Please upload an image or PDF."}, indent=4
        )
        return error_json, gr.Button(interactive=False)

    image_paths_to_process = []
    unsupported_files = []
    for file_obj in uploaded_files:
        file_path = file_obj.name
        temp_files_to_clean.append(file_path)

        if file_path.lower().endswith(".pdf"):
            pdf_page_images = convert_pdf_to_images(file_path)
            image_paths_to_process.extend(pdf_page_images)
            temp_files_to_clean.extend(pdf_page_images)
        elif file_path.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
        ):
            image_paths_to_process.append(file_path)
        else:
            unsupported_files.append(os.path.basename(file_path))

    if unsupported_files:
        unsupported_str = ", ".join(unsupported_files)
        results.append(
            json.dumps(
                {
                    "error": f"Unsupported file type(s) were ignored: {unsupported_str}",
                    "details": "Please upload only images (PNG, JPG, etc.) or PDF files.",
                },
                indent=4,
            )
        )

    for image_file in image_paths_to_process:
        try:
            image_path, width, height = array_to_image_path(image_file)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                            "resized_height": height,
                            "resized_width": width,
                        },
                        {"type": "text", "text": json_prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            generated_ids = model.generate(**inputs, max_new_tokens=4096)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            raw_output = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            raw_text = raw_output[0]

            try:
                start_index = raw_text.find("{")
                end_index = raw_text.rfind("}") + 1
                if start_index != -1 and end_index != 0:
                    json_string = raw_text[start_index:end_index]
                    parsed_json = json.loads(json_string)
                    parsed_json["source_page"] = os.path.basename(image_path)
                    formatted_json = json.dumps(parsed_json, indent=4)
                    results.append(formatted_json)
                else:
                    results.append(
                        f'{{"error": "Model did not return valid JSON.", "source_page": "{os.path.basename(image_path)}", "raw_response": "{raw_text}"}}'
                    )
            except json.JSONDecodeError:
                results.append(
                    f'{{"error": "Failed to decode JSON.", "source_page": "{os.path.basename(image_path)}", "raw_response": "{raw_text}"}}'
                )
        except Exception as e:
            results.append(
                f'{{"error": "An unexpected error occurred during processing.", "details": "{str(e)}"}}'
            )

    for f in temp_files_to_clean:
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError as e:
                print(f"Error deleting file {f}: {e}")

    final_json = "\n---\n".join(results)
    is_error = '"error":' in final_json
    return final_json, gr.Button(interactive=not is_error)


@spaces.GPU
def generate_explanation(json_text):
    if not json_text or '"error":' in json_text:
        return "Cannot generate an explanation. Please produce a valid JSON output first. 🙁"

    explanation_prompt = (
        "You are an expert data analyst. Your task is to provide a comprehensive, human-readable explanation "
        "of the following JSON data, which may represent one or more pages from a document. First, provide a textual explanation. "
        "If the JSON contains data from multiple sources (pages), explain each one. Then, if the JSON data represents a table, "
        "a list of items, or a receipt, you **must** re-format the key information into a Markdown table for clarity.\n\n"
        f"JSON Data:\n```json\n{json_text}\n```"
    )

    messages = [{"role": "user", "content": explanation_prompt}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], return_tensors="pt").to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    explanation_output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    return explanation_output


# Define the Gradio UI
css = """
  .gradio-container { font-family: 'IBM Plex Sans', sans-serif; }

  /* Default (Light Mode) Styles */
  #output-code, #output-code pre, #output-code code {
    background-color: #f0f0f0;
    border: 1px solid #e0e0e0;
    border-radius: 7px;
    color: #333;
  }
  #output-code .token.punctuation { color: #393a34; }
  #output-code .token.property, #output-code .token.string { color: #0b7500; }
  #output-code .token.number { color: #2973b7; }
  #output-code .token.boolean { color: #9a050f; }

  #explanation-box {
    min-height: 200px;
    border: 1px solid #e0e0e0;
    padding: 15px;
    border-radius: 7px;
  }

  /* Dark Mode Overrides targeting Gradio's .dark class */
  .dark #output-code, .dark #output-code pre, .dark #output-code code {
    background-color: #2b2b2b !important;
    border: 1px solid #444 !important;
    color: #f0f0f0 !important;
  }
  .dark #explanation-box {
    border: 1px solid #444 !important;
  }
  .dark #output-code code span {
     color: #f0f0f0 !important;
  }
  .dark #output-code .token.punctuation { color: #ccc !important; }
  .dark #output-code .token.property, .dark #output-code .token.string { color: #90ee90 !important; }
  .dark #output-code .token.number { color: #add8e6 !important; }
  .dark #output-code .token.boolean { color: #f08080 !important; }
"""

with gr.Blocks(theme=custom_theme, css=css) as demo:
    gr.Markdown("# Sparrow Qwen2-VL-7B Vision AI 👁️")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            # input_files = gr.Files(label="Upload Images or PDFs")
            input_files = gr.Files(
                label="Upload Images or PDFs",
                file_types=["image", ".pdf", ".png", ".jpg", ".jpeg", ".webp"],
            )
            text_input = gr.Textbox(
                label="Your Query",
                placeholder="e.g., Extract the total amount from this receipt.",
            )
            submit_btn = gr.Button("Analyze File(s)", variant="primary")

        with gr.Column(scale=2):
            output_text = gr.Code(
                label="Full JSON Response",
                language="json",
                elem_id="output-code",
                interactive=False,
            )
            explanation_btn = gr.Button(
                "📄 Generate Detailed Explanation", interactive=False
            )
            explanation_output = gr.Markdown(
                label="Detailed Explanation", elem_id="explanation-box"
            )

    submit_btn.click(
        fn=run_inference,
        inputs=[input_files, text_input],
        outputs=[output_text, explanation_btn],
        api_name="analyze_document",
    )

    explanation_btn.click(
        fn=generate_explanation,
        inputs=[output_text],
        outputs=[explanation_output],
        show_progress="full",
        api_name="generate_explanation",
    )

demo.queue()
# demo.launch(debug=True, share=True)
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
