from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import pipeline
import torch
from PIL import Image
import os

# Specify the directory where Tesseract OCR is installed
tesseract_path = r'C:\Program Files\Tesseract-OCR'

# Get the current value of the PATH environment variable
current_path = os.environ.get('PATH', '')

# Add the Tesseract directory to the PATH if it's not already there
if tesseract_path not in current_path:
    os.environ['PATH'] = f"{current_path};{tesseract_path}"

print("Tesseract OCR directory has been added to the PATH environment variable.")



# Load the image captioning model
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


def image_to_text(image_path):
    # Perform OCR to convert the image to text
    # You can use any OCR library such as pytesseract or Tesseract OCR
    # For simplicity, I'll assume you have pytesseract installed
    try:
        import pytesseract
        text = pytesseract.image_to_string(Image.open(image_path))
        return text
    except ImportError:
        print("pytesseract is not installed. Please install it using 'pip install pytesseract'.")
        return None

from transformers import pipeline

# Load the sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

def predict_caption_with_sentiment(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)

    pixel_values = feature_extractor(
        images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    # Analyze sentiment for each caption
    sentiment_scores = []
    for caption in preds:
        sentiment = sentiment_analyzer(caption)
        sentiment_scores.append(sentiment[0]['label'])

    print("Final Captions and Sentiments:")
    for i, pred in enumerate(preds):
        print(f"Caption: {pred}, Sentiment: {sentiment_scores[i]}")

    return preds, sentiment_scores


# Run prediction with sentiment analysis
predict_caption_with_sentiment(['anitha.JPG'])