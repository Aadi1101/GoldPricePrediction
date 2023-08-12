from flask import Flask,render_template
import gradio as gr
import numpy as np
import dill

def generate_output(input_text:str):
    # input_text = SPX_USO_SLV_EUR_USD_comma_separated_values
    output_text = [float(item) for item in input_text.split(',')]
    with open('models/preprocessor.pkl', 'rb') as p:
        preprocessor = dill.load(p)
    output_text = np.array(output_text).reshape(1, -1)
    output_text_dims = preprocessor.transform(output_text)
    with open('models/best_model.pkl', 'rb') as m:
        model = dill.load(m)
    GLD = model.predict(output_text_dims)
    return GLD

gr_interface = gr.Interface(
    fn=generate_output,
    inputs=gr.inputs.Textbox(placeholder="Enter text here",label="SPX,USO,SLV,EUR/USD(comma_separated_values)"),
    outputs=gr.outputs.Textbox(label="GLD"),
    title="Gold Price Prediction"
)

app=Flask('__name__')
@app.route('/')
def index():
    return render_template('index.html')

gr_interface.launch(share=True)

if __name__ == '__main__':
    app.run()