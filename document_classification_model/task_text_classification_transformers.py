# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="nlpodyssey/bert-italian-uncased-iptc-headlines")

# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("nlpodyssey/bert-italian-uncased-iptc-headlines")
model = AutoModelForSequenceClassification.from_pretrained("nlpodyssey/bert-italian-uncased-iptc-headlines")