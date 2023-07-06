# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModel
import gradio as gr

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-chat-7b",
                                          trust_remote_code=True)
# model = AutoModel.from_pretrained("internlm/internlm-chat-7b",
#                                   trust_remote_code=True,
#                                   device='cuda')
model = AutoModel.from_pretrained("internlm/internlm-chat-7b",
                                  trust_remote_code=True)


def generate(prompt):
    model = model.eval()
    response, history = model.chat(tokenizer, prompt, history=history)
    return response


input_component = gr.Textbox(label="Prompt")
output_component = gr.Textbox(label="output")
examples = [["今天天气如何？"], ["你是谁？"]]
description = "浦语在线体验1.0"
gr.Interface(generate,
             inputs=input_component,
             outputs=output_component,
             examples=examples,
             title="InternLM Chat demo v1.0",
             description=description).launch()
