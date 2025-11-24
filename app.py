import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import glob


# --- 1. Dynamic Model Loading Function ---
def get_latest_model_path():
    """
    Automatically finds the most recently modified 'best.pt' file 
    inside the 'models' directory, including subdirectories.
    """
    # Search for all files named 'best.pt' inside 'models' folder
    # recursive=True allows looking inside subfolders like yolov8_drone_bird10/weights/
    search_pattern = os.path.join("models", "**", "weights", "best.pt")
    possible_models = glob.glob(search_pattern, recursive=True)
    
    if not possible_models:
        return None
    
    # Sort found models by modification time (newest first) and pick the first one
    latest_model = max(possible_models, key=os.path.getmtime)
    return latest_model

# --- 2. Application Setup ---
st.set_page_config(
    page_title="Aerial Surveillance AI",
    page_icon="🚁",
    layout="wide"
)

def main():
    # --- 3. Professional UI: Sidebar ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3069/3069172.png", width=100)
        st.title("Project Config")
        st.info(
            """
            **Project:** Aerial Object Classification
            
            """
        )
        st.divider()
        st.write("### ⚙️ Settings")
        confidence = st.slider("Detection Confidence", 0.1, 1.0, 0.25, 0.05)
        st.caption("Lower confidence detects more objects but may increase false positives.")

    # --- Main Content ---
    st.title("🚁 Aerial Object Classification & Detection")
    st.markdown(
        """
        <style>
        .big-font { font-size:18px !important; }
        </style>
        <p class="big-font">
        Upload aerial imagery to automatically identify and locate <b>Birds</b> and <b>Drones</b> 
        to ensure airspace safety and wildlife protection.
        </p>
        """, unsafe_allow_html=True
    )

    st.divider()

    # --- Load Model Dynamically ---
    model_path = get_latest_model_path()
    
    if model_path:
        # Load the model
        try:
            model = YOLO(model_path)
            # Show which model is being used (helpful for debugging)
            st.sidebar.success(f"Loaded Model:\n{os.path.basename(os.path.dirname(os.path.dirname(model_path)))}")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return
    else:
        st.error("❌ No trained model found! Please run 'train_yolo.py' first.")
        st.stop()

    # --- Image Upload & Processing ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Source Image", use_container_width=True)

    with col2:
        st.subheader("🎯 Detection Results")
        
        if uploaded_file:
            with st.spinner("Analyzing scene..."):
                # Run inference
                results = model.predict(image, conf=confidence, device='cpu')
                
                # Visualize results
                for r in results:
                    im_array = r.plot()  # plot() returns a BGR numpy array
                    im = Image.fromarray(im_array[..., ::-1])  # RGB conversion
                    st.image(im, caption="AI Analysis", use_container_width=True)
                    
                    # Statistics
                    st.success("Analysis Complete")
                    
                    # Count classes
                    counts = {}
                    for cls in r.boxes.cls:
                        name = model.names[int(cls)]
                        counts[name] = counts.get(name, 0) + 1
                    
                    # Display counts cleanly
                    if counts:
                        st.write("### Objects Found:")
                        for name, count in counts.items():
                            st.write(f"- **{count}** {name}(s)")
                    else:
                        st.warning("No objects detected at this confidence level.")

        else:
            # Placeholder when no image is uploaded
            st.info("Waiting for upload...")
            st.markdown(
                """
                **How it works:**
                1. Upload an image (JPG/PNG).
                2. The AI scans for birds or drones.
                3. Bounding boxes are drawn around detected objects.
                """
            )

if __name__ == "__main__":
    main()
    