import gradio as gr
from transformers import BartTokenizer, BartForConditionalGeneration
import PyPDF2

# Load BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Function to summarize PDF text
def summarize_pdf(pdf_file):
    # Read the PDF file
    pdf_reader = PyPDF2.PdfReader(pdf_file.name)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Summarize the text
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Interface for Gradio
iface = gr.Interface(
    fn=summarize_pdf,
    inputs="file",
    outputs="text",
    title="PDF Summarizer",
    description="Upload a PDF file, and the model will provide a summary.",
    live=True,
    capture_session=True,
)

# Launch the Gradio interface
iface.launch()
