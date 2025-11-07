# 🧠 Sign Language Recognition System  

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-ff6f00?logo=tensorflow)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Real%20Time%20Hand%20Tracking-orange?logo=google)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green?logo=opencv)
![Accuracy](https://img.shields.io/badge/Accuracy-97%25-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🎯 Project Overview  

The **Sign Language Recognition System** leverages **Convolutional Neural Networks (CNN)** and **MediaPipe** to identify and interpret sign language gestures in real-time.  
This system aims to **bridge the communication gap** between mute and hearing individuals — allowing smoother and more inclusive interactions.  

---

## 🧩 Key Features  

✅ Real-time **hand detection and tracking** using **MediaPipe**  
✅ High-accuracy gesture classification via **CNN**  
✅ Dataset of **10,000+ hand sign images**  
✅ Supports **A–Z alphabet gestures**  
✅ Fast and efficient real-time predictions  

---

## 📊 Dataset Preparation  

- Collected **10,000 images** covering various hand signs for each alphabet letter.  
- Maintained **uniform backgrounds** and **consistent lighting** to ensure accuracy.  
- Split data:
  - 🧠 **80%** → Training  
  - 🧪 **20%** → Testing  
- Applied **normalization** for better convergence and model stability.  

---

## ⚙️ Tech Stack  

| Category | Tools / Libraries |
|-----------|-------------------|
| Programming | Python |
| Deep Learning | TensorFlow / Keras |
| Hand Tracking | MediaPipe |
| Computer Vision | OpenCV |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib |

---

## 🧮 Model Workflow  

1. **Palm Detection & Landmark Extraction**  
   → Using **MediaPipe Hands** for precise hand landmark detection.  

2. **Feature Extraction via CNN**  
   → Convolutional layers capture essential gesture features.  

3. **Training & Validation**  
   → Achieved **97% accuracy** with optimized hyperparameters.  

---

## 🚀 Results  

- 🟢 **Accuracy:** 97% on test data  
- ⚡ **Real-Time Prediction:** Low latency  
- ✋ **Robust Recognition:** Works across lighting & hand orientations  

---

## 🌱 Future Enhancements  

- 🔊 Integrate **text-to-speech output** for gesture-to-voice translation  
- 🤖 Support **multi-hand** and **word-level recognition**  
- ☁️ Deploy on **web / mobile platforms**  
- 🧩 Extend dataset for **numbers and symbols**  

---

## 👩‍💻 Author  

**Project:** Sign Language Recognition System  
**Developer:** Rohan Patil.  
**Core Technologies:** CNN, MediaPipe, TensorFlow, OpenCV  
**Model Accuracy:** 97%  

---

## 🪴 License  

This project is licensed under the **MIT License** — free to use, modify, and improve with proper credit.

---

> 💡 _“Breaking Barriers — Empowering Communication through AI and Sign Recognition.”_
