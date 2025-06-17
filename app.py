import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
from Q1RAM import Q1RAM
from QIP import FRQI_encoding, decode_frqi_image_aer,group_dict_by_prefix

# --- Helper Functions ---
def encode_image(source_image):
  h,w=source_image.shape
  n_w=int(np.ceil(np.log2(w)))
  n_h=int(np.ceil(np.log2(h)))
  qc= QuantumCircuit()
  qr_position,qr_color=FRQI_encoding(qc,source_image,apply_h=True)

  qc_tc=qc.decompose().decompose().decompose()
  qc_tc.save_statevector()
  
  simulator=AerSimulator(method="statevector")
  zero_threshold=0.05/(2**(2*len(qr_position)+len(qr_color)))
  statevector=simulator.run(qc_tc).result().get_statevector(qc_tc)
  probabilities=statevector.probabilities_dict()
  result_image=decode_frqi_image_aer(probabilities,n_w)
  return result_image

def write_image(source_image):
    w, h = source_image.shape
    n_w = int(np.ceil(np.log2(w)))
    n_h = int(np.ceil(np.log2(h)))

    qc = QuantumCircuit()
    qr_position, qr_color = FRQI_encoding(qc, source_image, apply_h=True)

    qr_address_bus = QuantumRegister(1, name="address")
    qc.add_register(qr_address_bus)
    qr_data_bus = [*qr_position, *qr_color]

    qram = Q1RAM(len(qr_address_bus), len(qr_data_bus), qc, qr_address_bus=qr_address_bus, qr_data_bus=qr_data_bus)
    qram.apply_write()

    qc_tc = qc.decompose().decompose().decompose()
    qc_tc.save_statevector()

    simulator = AerSimulator(method="statevector")
    zero_threshold = 0.05 / (2 ** (2 * len(qr_position) + len(qr_data_bus)))
    statevector = simulator.run(qc_tc).result().get_statevector(qc_tc)
    probabilities = statevector.probabilities_dict(qram.qr_data_register_index+qram.qr_address_register_index )

    correction_factor = 2 ** (2 * len(qr_address_bus))
    probabilities = {k: v * correction_factor for k, v in probabilities.items() if v > zero_threshold}

    images_probs = group_dict_by_prefix(probabilities, len(qr_address_bus))
    return {k: decode_frqi_image_aer(probs, n_w) for k, probs in images_probs.items()}

# --- Streamlit UI with Session State ---
st.set_page_config(page_title="QRAM Image Demo", layout="centered")
# st.title("Quantum Image Processing with QRAM")

# Init session state
if "source_image" not in st.session_state:
    st.session_state.source_image = None
if "encoded_image" not in st.session_state:
    st.session_state.encoded_image = None
if "qram_images" not in st.session_state:
    st.session_state.qram_images = None

st.image("q1ram_logo.jpg", width=200)
st.title("Quantum Image Processing with QRAM")
st.write("This demo showcases how to encode a grayscale image using FRQI encoding and store it in QRAM. "
         "You can upload a grayscale image, encode it, and then write it to QRAM for further processing.")

# Step 1: Upload image
st.subheader("Step 1: Upload Grayscale Image")
st.image("step1.png")
uploaded_file = st.file_uploader("Step 1: Upload grayscale image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    from PIL import Image
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((64, 64))  # Resize to 32x32 pixels
    image_np = np.array(image)
    st.session_state.source_image = image_np
    col1,col2,col3 =st.columns(3)

    with col2:
        st.image(image_np, caption="Uploaded Image (64x64)")

# Step 2: Encode Image
st.subheader("Step 2: Encode Image")
st.image("step2.png")
if st.session_state.source_image is not None:
    if st.button("Step 2: Encode Image"):
        st.write(st.session_state.source_image.shape)
        st.session_state.encoded_image = encode_image(st.session_state.source_image)

    if st.session_state.encoded_image is not None:
        st.subheader("Decoded Image")
        col1,col2,col3 =st.columns([1,1,1])
        with col2:
            # st.image(st.session_state.encoded_image)
            fig = plt.figure(figsize=(1, 1))
            fig, ax = plt.subplots()
            ax.imshow(st.session_state.encoded_image, cmap='gray')
            ax.axis('off')
            st.pyplot(fig)

# Step 3: Write to QRAM
st.header("Step 3: Write to QRAM and Decode")
st.image("step3.png")
if st.session_state.source_image is not None:
    if st.button("Step 3: Write to QRAM and Decode"):
        st.session_state.qram_images = write_image(st.session_state.source_image)

    if st.session_state.qram_images:
        st.subheader("Decoded Images from QRAM")
        num_images = len(st.session_state.qram_images)
        fig = plt.figure(figsize=(2 * num_images, 2))
        for i, (k, v) in enumerate(st.session_state.qram_images.items()):
            plt.subplot(1, num_images, i + 1)
            plt.title(f"Image at address {k}")
            plt.imshow(v, cmap='gray')
            plt.axis('off')
        st.pyplot(fig)
