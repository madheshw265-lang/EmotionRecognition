<div align="center">

<img src="assets/banner.png" alt="Real-Time Facial Emotion Recognition" width="100%">

# 🎭 Real-Time Facial Emotion Recognition

### AI-Powered Facial Expression Classification Using Computer Vision & Deep Learning

<p>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/OpenCV-4.10-5C3EE8?logo=opencv&logoColor=white">
  <img src="https://img.shields.io/badge/TensorFlow-2.21-FF6F00?logo=tensorflow&logoColor=white">
  <img src="https://img.shields.io/badge/Keras-3.15-D00000?logo=keras&logoColor=white">
  <img src="https://img.shields.io/badge/NumPy-2.5-013243?logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/Deep%20Learning-CNN-8A2BE2">
  <img src="https://img.shields.io/badge/Computer%20Vision-AI-0078D4">
  <img src="https://img.shields.io/badge/Real--Time-Detection-00A86B">
</p>

</div>

---

## 📖 About

**Real-Time Facial Emotion Recognition** is a computer vision and deep learning project that uses a webcam to detect a human face and classify visible facial-expression patterns in real time.

The system captures live video from the laptop's built-in camera, detects faces using OpenCV, preprocesses the detected facial region, and passes the processed image to a pretrained Convolutional Neural Network (CNN) model.

The model generates predictions for seven facial-expression categories. The application selects the highest-scoring category and displays the predicted expression together with a confidence score directly on the live camera feed.

This project demonstrates how **Artificial Intelligence, Computer Vision, Image Processing, and Deep Learning** can be integrated into a practical real-time application.

> **Important:** The system classifies visible facial-expression patterns. It does not determine a person's actual internal emotional, psychological, or mental state.

---

## 🎯 Problem Statement

Human-computer interaction traditionally relies on keyboards, touchscreens, and voice commands. However, human communication also contains important visual information through facial expressions.

Recognizing facial-expression patterns automatically is a challenging computer vision problem because expressions can vary between individuals and can be affected by lighting, camera quality, face orientation, and other environmental conditions.

This project addresses this problem by creating a real-time system that:

- Captures video from a webcam
- Detects faces automatically
- Extracts facial regions
- Preprocesses facial images
- Uses deep learning for expression classification
- Displays the predicted expression in real time
- Provides a prediction confidence score

---

## 🎯 Project Objectives

The main objectives of this project are:

- Build a real-time facial-expression recognition system.
- Use a laptop webcam as the primary input device.
- Detect human faces automatically.
- Process facial images using computer vision techniques.
- Apply a pretrained CNN model for facial-expression classification.
- Display predictions directly on the live camera feed.
- Display the confidence score associated with the prediction.
- Demonstrate practical integration of OpenCV and TensorFlow.
- Build a simple and extensible AI project for learning and demonstration.

---

## ✨ Key Features

### 🎥 Real-Time Webcam Processing

The application continuously captures frames from the laptop's built-in webcam.

### 👤 Face Detection

Faces are detected automatically using OpenCV's Haar Cascade classifier.

### 🧠 Deep Learning Classification

A pretrained CNN model is used to classify facial-expression patterns.

### 📊 Confidence Score

The application displays the prediction score for the selected expression.

### ⚡ Real-Time Inference

The model processes facial images continuously as the camera captures new frames.

### 🖼️ Image Preprocessing

Detected faces are converted to grayscale, resized to 48 × 48 pixels, and normalized before model inference.

### 💻 Laptop-Based System

The project can be demonstrated using a standard laptop and its built-in webcam.

### 🔧 Modular Project Structure

The project separates camera testing, face detection, and emotion recognition into different Python programs.

---

## 😊 Supported Facial Expressions

The model provides seven output categories:

| # | Expression | Meaning |
|---|---|---|
| 1 | 😠 **Angry** | Facial patterns associated with anger |
| 2 | 🤢 **Disgust** | Facial patterns associated with disgust |
| 3 | 😨 **Fear** | Facial patterns associated with fear |
| 4 | 😊 **Happy** | Facial patterns associated with happiness |
| 5 | 😢 **Sad** | Facial patterns associated with sadness |
| 6 | 😲 **Surprise** | Facial patterns associated with surprise |
| 7 | 😐 **Neutral** | No strong expression detected |

The model generates scores for all seven categories and the application displays the category with the highest score.

---

## 🏗️ System Architecture

The system follows a complete real-time computer vision and deep learning pipeline:

**Webcam → Video Frame → Face Detection → Face Extraction → Image Preprocessing → CNN Model → Prediction → Confidence Score → Display**

### Architecture Flow

```text
┌─────────────────────────────┐
│       Laptop Webcam         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       Video Capture         │
│           OpenCV            │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       Face Detection        │
│       Haar Cascade          │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│      Face Extraction        │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│    Image Preprocessing      │
│                             │
│    Grayscale                │
│    Resize: 48 × 48          │
│    Normalize: 0–1           │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│      CNN Model              │
│     TensorFlow/Keras        │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│    Seven Class Prediction   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Highest Score Selected      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Expression + Confidence     │
│       Displayed             │
└─────────────────────────────┘
