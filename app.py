import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model("fruit_model.h5")

classes = [
    "Apple",
    "Banana",
    "avocado",
    "cherry",
    "kiwi",
    "mango",
    "orange",
    "pinenapple",
    "strawberries",
    "watermelon"
]

def predict_fruit(image):
    image = image.resize((224,224))
    img = np.array(image)/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)

    return classes[np.argmax(pred)]

app = gr.Interface(
    fn=predict_fruit,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Fruit Recognition App"
)

app.launch()
