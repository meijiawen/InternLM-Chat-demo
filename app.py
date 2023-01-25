from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr

tokenizer = AutoTokenizer.from_pretrained("merve/chatgpt-prompts-bart-long")
model = AutoModelForSeq2SeqLM.from_pretrained("merve/chatgpt-prompts-bart-long", from_tf=True)

def generate(prompt):

    batch = tokenizer(prompt, return_tensors="pt")
    generated_ids = model.generate(batch["input_ids"], max_new_tokens=150)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return output[0]

input_component = gr.Textbox(label = "Input a persona, e.g. photographer", value = "photographer")
output_component = gr.Textbox(label = "Prompt")
examples = [["photographer"], ["developer"]]
description = "This app generates ChatGPT prompts, it's based on a BART model trained on [this dataset](https://huggingface.co/datasets/fka/awesome-chatgpt-prompts). Simply enter a persona that you want the prompt to be generated based on."
gr.Interface(generate, inputs = input_component, outputs=output_component, title = "ChatGPT Prompt Generator", description=description).launch()
