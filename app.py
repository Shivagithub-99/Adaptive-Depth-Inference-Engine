import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np

# ---------------- Model ----------------
class SmartNet(nn.Module):
    def __init__(self):
        super(SmartNet, self).__init__()

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x, threshold):
        confidences = []

        # Layer 1
        x = F.relu(self.fc1(x))
        out1 = F.softmax(nn.Linear(128, 10)(x), dim=1)
        conf1 = torch.max(out1).item()
        confidences.append(conf1)

        if conf1 >= threshold:
            return out1, 1, confidences

        # Layer 2
        x = F.relu(self.fc2(x))
        out2 = F.softmax(nn.Linear(64, 10)(x), dim=1)
        conf2 = torch.max(out2).item()
        confidences.append(conf2)

        if conf2 >= threshold:
            return out2, 2, confidences

        # Final Layer
        out3 = F.softmax(self.fc3(x), dim=1)
        conf3 = torch.max(out3).item()
        confidences.append(conf3)

        return out3, 3, confidences


# ---------------- Fixed Depth ----------------
def fixed_depth(model):
    energy = 0
    start = time.time()

    x = torch.randn(1, 128)

    x = F.relu(model.fc1(x)); energy += 10
    x = F.relu(model.fc2(x)); energy += 10
    x = model.fc3(x); energy += 10

    latency = round(time.time() - start, 3)
    return latency, energy


# ---------------- Smart Depth ----------------
def smart_depth(model, battery):
    if battery < 30:
        threshold = 0.3
    elif battery < 60:
        threshold = 0.5
    else:
        threshold = 0.7

    start = time.time()

    x = torch.randn(1, 128)

    output, exit_layer, confidences = model(x, threshold)
    energy = exit_layer * 8

    latency = round(time.time() - start, 3)
    return latency, energy, exit_layer, confidences, threshold


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="SMART DEPTH AI", layout="centered")

st.title("🔍 SMART DEPTH – Adaptive Neural Network")
st.write("Energy-efficient inference using early-exit strategy")

battery = st.slider("🔋 Battery Level (%)", 10, 100, 50)

model = SmartNet()

if st.button("🚀 Run Inference"):
    with st.spinner("Running model..."):

        fd_latency, fd_energy = fixed_depth(model)
        sd_latency, sd_energy, exit_layer, confs, thresh = smart_depth(model, battery)

    st.success("Execution Completed")

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Exit Layer", exit_layer)
    col2.metric("Threshold", thresh)
    col3.metric("Battery (%)", battery)

    st.subheader("⚡ Performance")

    st.write(f"**Fixed Depth** → Latency: `{fd_latency}s`, Energy: `{fd_energy}`")
    st.write(f"**Smart Depth** → Latency: `{sd_latency}s`, Energy: `{sd_energy}`")

    # Energy Chart
    fig1, ax1 = plt.subplots()
    ax1.bar(["Fixed", "Smart"], [fd_energy, sd_energy])
    ax1.set_title("Energy Comparison")
    st.pyplot(fig1)

    # Confidence Chart
    fig2, ax2 = plt.subplots()
    layers = list(range(1, len(confs) + 1))
    ax2.plot(layers, confs, marker='o')
    ax2.axhline(thresh, linestyle='--')
    ax2.set_title("Confidence vs Layers")
    st.pyplot(fig2)