# NLLB-200-600M-fine-tuning

---
## nllb-200-600M-En-Ar

This model is a fine-tuned version of the NLLB-200-600M model, specifically adapted for translating from English to Egyptian Arabic. Fine-tuned on a custom dataset of 12,000 samples, it aims to provide high-quality translations that capture the nuances and colloquial expressions of Egyptian Arabic.

The dataset used for fine-tuning was collected from high-quality transcriptions of videos, ensuring the language data is rich and contextually accurate.
### Model Details

- **Base Model**: [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)
- **Language Pair**: English to Egyptian Arabic
- **Dataset**: 12,000 custom translation pairs


### Streamlit App
can try the model with streamlit app based on huggingface space from here: [Streamlit App](https://mhassanen-nllb-en-ar-translation.hf.space/)


### Usage

To use this model for translation, you can load it with the `transformers` library:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "Mhassanen/nllb-200-600M-En-Ar"
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn", tgt_lang="arz_Arab")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def translate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated_tokens = model.generate(**inputs)
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return translated_text

text = "Hello, how are you?"
print(translate(text))
```


### Performance

The model has been evaluated on a validation set to ensure translation quality. While it excels at capturing colloquial Egyptian Arabic, ongoing improvements and additional data can further enhance its performance.

### Limitations

- **Dataset Size**: The custom dataset consists of 12,000 samples, which may limit coverage of diverse expressions and rare terms.
- **Colloquial Variations**: Egyptian Arabic has many dialectal variations, which might not all be covered equally.

### Acknowledgements

This model builds upon the NLLB-200-600M developed by Facebook AI, fine-tuned to cater specifically to the Egyptian Arabic dialect.

Feel free to contribute or provide feedback to help improve this model!

