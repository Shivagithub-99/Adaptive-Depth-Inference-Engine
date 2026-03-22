import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt

# ---------------- Utility Function ----------------
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# ---------------- Fixed Depth Inference ----------------
def fixed_depth():
    energy = 0
    start = time.time()
    features = np.random.rand(128)

    for _ in range(6):
        features = np.tanh(features @ np.random.rand(128, 128))
        energy += 10
        time.sleep(0.2)

    latency = round(time.time() - start, 3)
    return latency, energy

# ---------------- Smart Depth Inference ----------------
def smart_depth(battery):
    if battery < 30:
        threshold = 0.70
    elif battery < 60:
        threshold = 0.80
    else:
        threshold = 0.90

    confidences = []
    energy = 0
    start = time.time()
    features = np.random.rand(128)

    for layer in range(1, 7):
        features = np.tanh(features @ np.random.rand(128, 128))
        conf = np.max(softmax(np.random.rand(5)))
        confidences.append(conf)
        energy += 8
        time.sleep(0.2)

        if conf >= threshold:
            exit_layer = layer
            break
    else:
        exit_layer = 6

    latency = round(time.time() - start, 3)
    return latency, energy, exit_layer, confidences, threshold

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="SMART DEPTH Execution", layout="centered")

st.title("🔍 SMART DEPTH – Adaptive Inference System")
st.write("Energy-aware early-exit inference with real-time visualization")

battery = st.slider("🔋 Battery Level (%)", 10, 100, 40)

if st.button("🚀 Run Inference"):
    with st.spinner("Running inference..."):
        fd_latency, fd_energy = fixed_depth()
        sd_latency, sd_energy, exit_layer, confs, thresh = smart_depth(battery)

    st.success("Execution Completed Successfully")

    # ---------------- Metrics ----------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Exit Layer", exit_layer)
    col2.metric("Threshold", thresh)
    col3.metric("Battery (%)", battery)

    st.subheader("⚡ Performance Summary")
    st.write(f"**Fixed Depth** → Latency: `{fd_latency}s`, Energy: `{fd_energy}`")
    st.write(f"**Smart Depth** → Latency: `{sd_latency}s`, Energy: `{sd_energy}`")

    st.info("📊 Visual analysis of adaptive inference behavior")

    # ---------------- Energy Comparison Chart ----------------
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    bars = ax1.bar(
        ["Fixed Depth", "Smart Depth"],
        [fd_energy, sd_energy]
    )

    ax1.set_title("Energy Consumption Comparison")
    ax1.set_ylabel("Energy Units")
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height}",
            ha="center",
            fontsize=10
        )

    st.pyplot(fig1)

    # ---------------- Confidence vs Layers Chart ----------------
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    layers = list(range(1, len(confs) + 1))

    ax2.plot(
        layers,
        confs,
        marker="o",
        linewidth=2,
        label="Confidence"
    )

    ax2.axhline(
        thresh,
        linestyle="--",
        label="Confidence Threshold"
    )

    ax2.axvline(
        exit_layer,
        linestyle=":",
        label=f"Early Exit @ Layer {exit_layer}"
    )

    ax2.set_title("Confidence Progression Across Network Layers")
    ax2.set_xlabel("Network Layers")
    ax2.set_ylabel("Confidence Score")
    ax2.set_ylim(0, 1)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    st.pyplot(fig2)
