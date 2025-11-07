# 🧠 Sign Language Recognition System

A deep learning-based **Sign Language Recognition System** that utilizes **Convolutional Neural Networks (CNN)** and **MediaPipe** to detect and interpret sign language gestures in real-time — bridging the communication gap between mute and hearing individuals.

---

## 🎯 Project Overview

The goal of this project is to create an intelligent system that can accurately recognize and translate **hand gestures** from sign language into readable or audible text.  
Many people are not familiar with sign language, making communication with mute individuals challenging. This project provides a **technological bridge** to overcome that barrier.

---

## 🧩 Key Features

✅ Real-time **hand detection and tracking** using **MediaPipe**  
✅ High-accuracy gesture classification with **CNN architecture**  
✅ Dataset of **10,000+ labeled images** for training and testing  
✅ Supports **all alphabetic gestures (A–Z)**  
✅ Enables **seamless communication** between deaf and hearing individuals  

---

## 📊 Dataset Preparation

- Images of various **hand signs representing each alphabet letter** were captured.  
- Ensured **uniform background** and **consistent lighting** for better model accuracy.  
- Dataset split:
  - **80%** → Training
  - **20%** → Testing
- Normalization applied for optimal CNN performance.  

---

## ⚙️ Technologies Used

| Category | Tools / Libraries |
|-----------|-------------------|
| Programming | Python |
| Frameworks | TensorFlow / Keras |
| Hand Tracking | MediaPipe |
| Deep Learning | Convolutional Neural Networks (CNN) |
| Visualization | OpenCV, Matplotlib |
| Dataset Handling | NumPy, Pandas |

---

## 🧮 Model Implementation

1. **Palm Detection & Hand Landmark Extraction**  
   - Implemented using **MediaPipe Hands** to detect and track real-time hand landmarks.

2. **Feature Extraction with CNN**  
   - CNN trained on preprocessed image dataset for accurate gesture classification.

3. **Model Training & Validation**  
   - Achieved **97% accuracy** on test dataset.
   - Optimized model performance using batch normalization and dropout layers.

---

## 🚀 Results & Performance

- ✅ Achieved **97% overall accuracy**
- ✅ Real-time gesture prediction with low latency
- ✅ Robust to varying hand orientations and lighting conditions

---

## 🌱 Future Enhancements

- 🔊 Integrate **text-to-speech** for auditory output  
- 🤖 Add **multi-hand gesture** and **word-level recognition**  
- ☁️ Deploy as a **web or mobile application**  
- 🧩 Expand dataset for **numerical and symbolic gestures**

---

## 👩‍💻 Author

**Project Name:** Sign Language Recognition System  
**Developer:** Rohan P.  
**Core Technologies:** CNN, MediaPipe, TensorFlow, OpenCV  
**Accuracy:** 97%  

---

## 🪴 License

This project is open-source under the **MIT License** — free to use and modify with attribution.

---

> _"Breaking Barriers — Empowering Communication through AI and Sign Recognition."_
