
# ✋ Hand Gesture Recognition using Deep Learning  

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)  
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)  
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow?logo=scikitlearn)  
![Matplotlib](https://img.shields.io/badge/Matplotlib-Data%20Viz-lightblue?logo=plotly)  

---

## 📌 Project Overview  

This project is part of my **Prodigy Infotech Internship (Task 4)**.  
I developed a **Hand Gesture Recognition Model** that can classify different hand gestures from images using **Convolutional Neural Networks (CNNs)**.  

The model is trained on the **LeapGestRecog dataset** (Kaggle) and demonstrates how gesture recognition can be used for:  
- Human-Computer Interaction (HCI)  
- Gesture-based control systems  
- Interactive AI applications  

---

## 📂 Dataset  

**Dataset:** [LeapGestRecog (Kaggle)](https://www.kaggle.com/gti-upm/leapgestrecog)  

- Contains **30,000+ images** of hand gestures.  
- Images are categorized into **10 gesture classes**, captured under different lighting and backgrounds.  
- Structure: Each gesture has its own folder, with multiple subject subfolders.  

---

## ⚙️ Tech Stack  

- **Python** 🐍  
- **TensorFlow/Keras** → Deep Learning (CNN, training)  
- **OpenCV** → Image handling  
- **Scikit-learn** → Metrics, confusion matrix  
- **Matplotlib & Seaborn** → Data visualization  

---

## 🚀 Steps in the Project  

1. **Import Libraries**  
   Installed and imported TensorFlow, OpenCV, Pandas, Matplotlib, Scikit-learn.  

2. **Load Dataset**  
   Used `image_dataset_from_directory` to efficiently load images with train/validation split.  

3. **Preprocessing**  
   - Resized images to `128x128`  
   - Normalized pixel values (`0-1`)  
   - Applied **Data Augmentation** (flip, rotate, zoom)  

4. **Model Building**  
   - Built a **CNN model** with Conv2D, MaxPooling, Flatten, Dense layers.  
   - Also experimented with **Transfer Learning (MobileNetV2)** for efficiency.  

5. **Model Training**  
   - Optimizer: `Adam`  
   - Loss: `SparseCategoricalCrossentropy`  
   - Metric: `Accuracy`  
   - Epochs: `10` (with EarlyStopping for efficiency)  

6. **Evaluation**  
   - Training vs Validation Accuracy Graph 📈  
   - Confusion Matrix 🔲  
   - Classification Report 📊  

7. **Prediction Demo**  
   - Tested the model with a **single image prediction**.  
   - Displayed actual image with predicted gesture label.  

---

## 📊 Results  

- Validation Accuracy: **~85-90%** (varies with run/epochs)  
- Confusion Matrix shows model performance across all gesture classes.  
- The model can successfully recognize multiple hand gestures.  

**Training vs Validation Accuracy Graph:**  

![Accuracy Graph](https://via.placeholder.com/600x300.png?text=Training+vs+Validation+Accuracy)  

**Confusion Matrix:**  

![Confusion Matrix](https://via.placeholder.com/500x400.png?text=Confusion+Matrix)  

---

## 🔮 Future Improvements  

- Extend model for **real-time gesture recognition** using webcam + OpenCV.  
- Deploy as a **web application** using Flask/Streamlit.  
- Fine-tune with **more advanced architectures (ResNet, EfficientNet)**.  

---

## 🙌 Acknowledgements  

- **Dataset:** [LeapGestRecog (Kaggle)](https://www.kaggle.com/gti-upm/leapgestrecog)  
- **Prodigy Infotech** – Internship Task 4  

---

## 📌 How to Run  

1. Clone this repo:  
   ```bash
   git clone https://github.com/your-username/hand-gesture-recognition.git
   cd hand-gesture-recognition


2. Install dependencies:

   ```bash
   pip install tensorflow opencv-python matplotlib scikit-learn seaborn
   ```

3. Run Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. Train and evaluate the model by running cells step by step.

---

✨ Built with passion during my **Prodigy Infotech Internship** 🚀

```
