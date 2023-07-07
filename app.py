# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModel
import gradio as gr

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-chat-7b",
#                                           trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b",
                                          trust_remote_code=True)
# model = AutoModel.from_pretrained("internlm/internlm-chat-7b",
#                                   trust_remote_code=True,
#                                   device='cuda')
# model = AutoModel.from_pretrained("internlm/internlm-chat-7b",
#                                   trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b",
                                  trust_remote_code=True).half().cuda()


def predict(input, history=None):
    if history is None:
        history = []
    response, history = model.chat(tokenizer, input, history)
    return history, history


with gr.Blocks() as demo:
    gr.Markdown('''## internlm-chat-7b -  demo
    demo of the [internlm-chat-7b](https://github.com/InternLM/InternLM) model
    ''')
    state = gr.State([])
    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=600)
    with gr.Row():
        with gr.Column(scale=4):
            txt = gr.Textbox(show_label=False,
                             placeholder="Enter text and press enter").style(
                                 container=False)
        with gr.Column(scale=1):
            button = gr.Button("Generate")
    txt.submit(predict, [txt, state], [chatbot, state])
    button.click(predict, [txt, state], [chatbot, state])
demo.queue().launch()

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
