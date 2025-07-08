import torch
import base64
from io import BytesIO
from PIL import Image
import streamlit as st
import time
import fitz  # PyMuPDF

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from olmocr.prompts import build_finetuning_prompt

# --- PDF Processing Functions ---

def render_pdf_to_base64png(pdf_path, page_number, target_longest_image_dim=1024):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    w, h = img.size
    scale = target_longest_image_dim / max(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode(), img

def get_anchor_text(pdf_path, page_number, target_length=4000):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number - 1)
    text = page.get_text()
    return text[:target_length]


# --- Load Model and Processor ---

@st.cache_resource
def load_model_and_processor():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "allenai/olmOCR-7B-0225-preview", torch_dtype=torch.bfloat16
    ).eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    device = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    model.to(device)
    return model, processor, device

model, processor, device = load_model_and_processor()

# --- Streamlit UI ---

st.set_page_config(page_title="olmOCR Streamlit", layout="wide")
st.title("📄 olmOCR PDF Extraction")
st.markdown("Upload a PDF, choose a page, and extract structured data using the Qwen2VL model.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    doc = fitz.open("temp.pdf")
    total_pages = len(doc)

    page_number = st.slider("Select page number", 1, total_pages, 1)

    if st.button("🔍 Run OCR on selected page"):
        with st.spinner("Processing..."):
            start_time = time.time()

            image_base64, main_image = render_pdf_to_base64png("temp.pdf", page_number)
            anchor_text = get_anchor_text("temp.pdf", page_number)
            prompt = build_finetuning_prompt(anchor_text)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    ],
                }
            ]

            chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = processor(
                text=[chat_text],
                images=[main_image],
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            output = model.generate(
                **inputs,
                temperature=0.8,
                max_new_tokens=300,
                num_return_sequences=1,
                do_sample=True,
            )

            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = output[:, prompt_len:]
            result = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

            st.success("✅ OCR Extraction Complete")
            st.image(main_image, caption="Rendered Page", use_column_width=True)
            st.subheader("📄 Anchor Text Snippet")
            st.text(anchor_text[:800])
            st.subheader("🧠 OCR Output")
            st.code(result, language="json")

            st.info(f"Total processing time: {time.time() - start_time:.2f}s")
