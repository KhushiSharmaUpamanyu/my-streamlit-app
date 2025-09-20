import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# Config
# -----------------------------
MODEL_DIR = "./xlm_roberta_model"
TOKENIZER_DIR = "./xlm_roberta_tokenizer"
BATCH_SIZE = 32
MAX_LENGTH = 128

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load model & tokenizer
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    model.to(device)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Multilingual Query→Category Relevance", layout="wide")
st.title("Multilingual Query → Category Relevance")
st.write(
    "Upload a CSV file with columns: `language`, `origin_query`, `category_path`. "
    "The model will predict relevance (0 = Not Relevant, 1 = Relevant)."
)

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        # Ensure required columns exist
        required_cols = ['origin_query', 'category_path']
        if not all(col in df.columns for col in required_cols):
            st.error(f"CSV must contain columns: {required_cols}")
        else:
            # -----------------------------
            # Inference in batches
            # -----------------------------
            all_preds = []
            for i in range(0, len(df), BATCH_SIZE):
                batch_queries = df['origin_query'][i:i+BATCH_SIZE].astype(str).tolist()
                batch_categories = df['category_path'][i:i+BATCH_SIZE].astype(str).tolist()

                encodings = tokenizer(
                    batch_queries,
                    batch_categories,
                    truncation=True,
                    padding=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt"
                )

                input_ids = encodings['input_ids'].to(device)
                attention_mask = encodings['attention_mask'].to(device)

                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    all_preds.extend(preds)

            # -----------------------------
            # Save predictions
            # -----------------------------
            df['predicted_label'] = all_preds

            st.write("Predictions:")
            st.dataframe(df)

            # Download CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")
