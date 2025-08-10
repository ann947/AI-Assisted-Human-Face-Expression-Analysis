# AI-Assisted Human Face Expression Analysis

A real-time facial expression recognition system designed for customer service and retail applications using deep learning techniques.

## Overview

This project implements an AI-powered facial expression recognition system that can identify seven universal emotions (happiness, sadness, anger, fear, surprise, disgust, and neutrality) in real-time. The system is specifically tailored for business environments to enhance customer experience through proactive emotion monitoring.

## Features

- **Real-time Emotion Detection**: Process facial expressions in under 50ms
- **CNN Architecture**: Custom convolutional neural network achieving 53.43% accuracy
- **Business Integration**: Ready-to-deploy system with Streamlit interface
- **Multi-model Support**: Comparison with classical ML approaches (Random Forest, SVM, KNN)
- **Scalable Architecture**: Supports up to 50 concurrent users per deployment

## Tech Stack

- **Deep Learning**: TensorFlow/Keras
- **Computer Vision**: OpenCV
- **Web Interface**: Streamlit
- **Database**: SQLite
- **Classical ML**: scikit-learn
- **Data Processing**: NumPy, Pandas

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ann947/AI-Assisted-Human-Face-Expression-Analysis.git

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Dataset

Trained on FER-2013 dataset containing 35,887 grayscale images (48x48 pixels) across 7 emotion categories.

## Performance

- **CNN Model**: 53.43% accuracy, 48.4ms processing time
- **Best Classical ML**: Random Forest at 47.00% accuracy
- **Real-time Capability**: Sub-second response for business applications

## Applications

- Customer service emotion monitoring
- Retail customer satisfaction tracking
- Real-time service quality assessment
- Automated alert systems for negative emotions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.