from fastapi import FastAPI
import dill
import numpy as np
import gradio as gr


app=FastAPI()
@app.get('/')
def read_main():
    return {"message":"this is your main app"}

def generate_output(input_text:str):
    # input_text = SPX_USO_SLV_EUR_USD_comma_separated_values
    output_text = [float(item) for item in input_text.split(',')]
    with open('src\models\preprocessor.pkl', 'rb') as p:
        preprocessor = dill.load(p)
    output_text = np.array(output_text).reshape(1, -1)
    output_text_dims = preprocessor.transform(output_text)
    with open('src\models\\best_model.pkl', 'rb') as m:
        model = dill.load(m)
    GLD = model.predict(output_text_dims)
    return GLD


gr_interface = gr.Interface(
    fn=generate_output,
    inputs=gr.components.Textbox(placeholder="Enter text here",label="SPX,USO,SLV,EUR/USD(comma_separated_values)"),
    outputs=gr.components.Textbox(label="GLD"),
    title="Gold Price Prediction"
)
app = gr.mount_gradio_app(app,gr_interface,path='/gradio')