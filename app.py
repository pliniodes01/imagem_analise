import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, render_template
from PIL import Image
import os
from train_model import generate_caption  # Importar a função generate_caption
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Criar a pasta uploads se não existir
if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return render_template('result.html', caption='No file part')

    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', caption='No selected file')

    if file:
        try:
            # Sanitize the filename
            filename = secure_filename(file.filename)
            image_path = os.path.join('uploads', filename)
            file.save(image_path)

            # Gerar a legenda
            caption = generate_caption(image_path)

            # Limpar o arquivo após uso
            os.remove(image_path)

            return render_template('result.html', caption=caption)
        except Exception as e:
            return render_template('result.html', caption=f"An error occurred: {e}")

if __name__ == '__main__':
    app.run(debug=True)
