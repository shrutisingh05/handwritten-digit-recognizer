# DigitAI – Handwritten Digit Recognizer

A deep learning application that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

The application allows users to upload handwritten digit images and view the predicted digit, prediction confidence, and probability distribution across all ten classes.

## Features

- Handwritten digit recognition from uploaded images.
- CNN model trained on the MNIST dataset.
- Image preprocessing, including grayscale conversion, cropping, resizing, and normalization.
- Prediction confidence display.
- Probability chart for digits 0–9.
- Interactive web interface built with Streamlit.

## Model Performance

| Metric | Result |
|---|---:|
| Training accuracy | 99.41% |
| Validation accuracy | 98.77% |
| Test accuracy | 98.42% |

*Results are from the trained model's evaluation on the MNIST dataset. Real-world image performance may vary.*

## Technologies Used

- Python
- TensorFlow and Keras
- NumPy
- Matplotlib
- Pandas
- Streamlit
- Pillow
- Convolutional Neural Networks (CNN)
- MNIST Dataset

## Project Structure

```text
handwritten-digit-recognizer/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── notebooks/
    ├── explore_mnist.ipynb
    └── digit_recognizer.keras
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd handwritten-digit-recognizer
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment (Windows)

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser at:

`http://localhost:8501`

## How It Works

1. The user uploads an image containing a handwritten digit.
2. The application preprocesses the image.
3. The processed image is converted into a 28 × 28 grayscale input.
4. The trained CNN analyzes the image.
5. The application displays the predicted digit, confidence, and class probabilities.

## Future Improvements

- Add a digital drawing canvas.
- Improve recognition across different handwriting styles.
- Expand image preprocessing capabilities.
- Deploy the application online.

## Author

Developed as a hands-on deep learning project to explore CNNs, image preprocessing, model training, and AI application development.