🚀 SMART DEPTH – Adaptive Neural Network Inference System
📌 Overview

SMART DEPTH is an energy-efficient AI system that dynamically adjusts neural network computation using an early-exit mechanism. Instead of processing all layers every time, the model exits early when sufficient confidence is achieved, reducing latency and energy consumption.

🎯 Objectives
Implement an adaptive inference system using neural networks
Reduce computational cost and energy usage
Maintain prediction accuracy with minimal processing
Demonstrate real-time AI optimization for edge devices

🧠 Methodology

The system uses a Feedforward Neural Network (FNN) with multiple fully connected layers:

Input is processed layer by layer
At each layer, confidence is calculated
If confidence ≥ threshold → early exit
Else → continue deeper layers

👉 Threshold is dynamically adjusted based on battery level.

⚙️ Technologies Used
Python
Streamlit
PyTorch
NumPy
Matplotlib

🔋 Key Features
✅ Real Neural Network (trained on dataset)
✅ Early-Exit Mechanism
✅ Battery-Based Adaptive Threshold
✅ Energy & Latency Comparison
✅ Interactive Web Interface

📊 System Workflow
Input data is fed into the neural network
Features are extracted layer by layer
Confidence is calculated at each stage
Model exits early if threshold is met
Outputs prediction with reduced computation

📈 Performance Metrics
Latency → Time taken for inference
Energy Consumption → Based on layers executed
Exit Layer → Indicates efficiency

⚠️ Limitations
Model is trained with limited epochs
Accuracy can be improved with more training
Early exit depends on threshold tuning

🔮 Future Enhancements
Use deep CNN models for better accuracy
Deploy on mobile/edge devices
Add real-time data input
Optimize using hardware-aware AI techniques

▶️ How to Run Locally
pip install streamlit torch torchvision matplotlib
streamlit run app.py

🌐 Deployment

The project is deployed using Streamlit Community Cloud.
👉https://adaptive-depth-inference-engine-aicamgxe6pjzkmhofpqpuo.streamlit.app/

👨‍💻 Authors
Shiva Goud
Siddhartha Makam
Varun Sai Naga

🎯 Conclusion

SMART DEPTH demonstrates how adaptive neural networks can significantly reduce computation without compromising performance, making it suitable for real-world energy-constrained environments.
