# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-chat-7b",
#                                           trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-chat-7b",
                                          trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("internlm/internlm-chat-7b",
                                             trust_remote_code=True).cuda()

# tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b",
#                                           trust_remote_code=True)
# model = AutoModel.from_pretrained("THUDM/chatglm-6b",
#                                   trust_remote_code=True).half().cuda()


def predict(input, history=None):
    if history is None:
        history = []
    response, history = model.stream_chat(tokenizer, input, history)
    return history, history


top_p = 1
temperature = 1


def predict_stream(input, top_p, temperature, history=[]):
    history = list(map(tuple, history))
    for response, updates in model.stream_chat(tokenizer,
                                               input,
                                               history,
                                               top_p=top_p,
                                               temperature=temperature):
        yield updates


def reset_textbox():
    return gr.update(value="")


with gr.Blocks() as demo:
    gr.Markdown(
        '''demo of the [internlm-chat-7b](https://github.com/InternLM/InternLM) model
    ''')
    state = gr.State([])
    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=600)
    with gr.Row():
        with gr.Column(scale=4):
            inputs = gr.Textbox(
                show_label=False,
                placeholder="请输入prompt并按回车发送").style(container=False)
        with gr.Column(scale=1):
            button = gr.Button("提交")
        with gr.Column(scale=1):
            clear = gr.Button("清除")

    inputs.submit(
        predict,
        [inputs, top_p, temperature, chatbot],
        [chatbot],
    )
    inputs.submit(reset_textbox, [], [inputs])

    button.click(
        predict,
        [inputs, top_p, temperature, chatbot],
        [chatbot],
    )
    button.click(reset_textbox, [], [inputs])

    clear.click(lambda: None, None, chatbot, queue=False)

    # txt.submit(predict, [txt, state], [chatbot, state])
    # button.click(predict, [txt, state], [chatbot, state])

demo.queue().launch()

# input_component = gr.Textbox(label="Prompt")
# output_component = gr.Textbox(label="output")
# examples = [["今天天气如何？"], ["你是谁？"]]
# description = "浦语在线体验1.0"
# gr.Interface(generate,
#              inputs=input_component,
#              outputs=output_component,
#              examples=examples,
#              title="InternLM Chat demo v1.0",
#              description=description).launch()
