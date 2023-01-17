import torch
import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration

st.title('要約')
st.markdown('長い文を要約します')

_num_beams = 4 
_no_repeat_ngram_size = 3
_length_penalty = 1
_min_length = 12
_max_length = 128
_early_stopping = True

col1, col2, col3 = st.columns(3)
_length_penalty = col1.number_input("要約する文の長さの制限", value=_length_penalty)
_min_length = col2.number_input("最小文字数", value=_min_length)
_max_length = col3.number_input("最大文字数", value=_max_length)

text = st.text_area('要約する文を入力してください')

def run_model(input_text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    input_text = str(input_text)
    input_text = ' '.join(input_text.split())
    input_tokenized = bart_tokenizer.encode(input_text, return_tensors='pt').to(device)
    summary_ids = bart_model.generate(input_tokenized,
    num_beams=_num_beams,
    no_repeat_ngram_size=_no_repeat_ngram_size,
    length_penalty=_length_penalty,
    min_length=_min_length,
    max_length=_max_length,
    early_stopping=_early_stopping)

    output = [bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    st.write('要約文')
    st.success(output[0])

if st.button('Submit'):
    run_model(text)



