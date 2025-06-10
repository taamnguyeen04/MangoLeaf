import streamlit as st
import os
import torch
import torch.nn as nn
from torchvision.models import resnet34
import numpy as np
import cv2
from PIL import Image
import tempfile
import time
import base64
import io

# Set page config with custom icon and wide layout
st.set_page_config(
    page_title="MangoLeaf AI | Disease Classifier",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://example.com/help',
        'Report a bug': "https://example.com/bug",
        'About': "# MangoLeaf AI - Advanced Disease Detection"
    }
)

# Disease information with emojis and more details
DISEASE_INFO = {
    "Anthracnose": {
        "emoji": "🍄",
        "description": "A fungal disease causing black spots and rotting.",
        "symptoms": "Dark, sunken lesions on leaves, stems, flowers, and fruits.",
        "treatment": "Apply fungicides containing copper or sulfur. Remove and destroy infected plant parts.",
        "prevention": "Ensure proper spacing between trees for air circulation. Avoid overhead watering.",
        "severity": "High"
    },
    "Bacterial Canker": {
        "emoji": "🦠",
        "description": "Bacterial infection that causes lesions and defoliation.",
        "symptoms": "Water-soaked lesions that turn brown and crack, oozing bacterial exudate.",
        "treatment": "Prune infected branches. Apply copper-based bactericides.",
        "prevention": "Disinfect pruning tools. Avoid wounding trees.",
        "severity": "Medium-High"
    },
    "Cutting Weevil": {
        "emoji": "🐛",
        "description": "Insect damage leading to irregular cuts on leaves.",
        "symptoms": "Notched leaf edges caused by adult weevils feeding.",
        "treatment": "Use insecticides or biological controls like nematodes.",
        "prevention": "Remove leaf litter where weevils overwinter.",
        "severity": "Medium"
    },
    "Die Back": {
        "emoji": "🍂",
        "description": "Drying of twigs and leaves from tip to base.",
        "symptoms": "Twigs die back from the tip, leaves turn brown and fall.",
        "treatment": "Prune affected branches and apply fungicides.",
        "prevention": "Maintain tree vigor with proper nutrition.",
        "severity": "High"
    },
    "Gall Midge": {
        "emoji": chr(0x1FAB0),
        "description": "Insect-caused galls leading to curling and swelling.",
        "symptoms": "Swollen, distorted leaves with small maggots inside.",
        "treatment": "Remove affected leaves. Use appropriate insecticides.",
        "prevention": "Monitor for early signs of infestation.",
        "severity": "Medium"
    },
    "Healthy": {
        "emoji": "✅",
        "description": "Leaf with no visible disease.",
        "symptoms": "Normal green color, uniform shape, no spots or deformities.",
        "treatment": "Maintain good cultural practices to prevent diseases.",
        "prevention": "Regular inspection and proper care.",
        "severity": "None"
    },
    "Powdery Mildew": {
        "emoji": "❄️",
        "description": "Fungal infection with white powdery patches.",
        "symptoms": "White powdery growth on leaves and shoots.",
        "treatment": "Apply sulfur or potassium bicarbonate-based fungicides.",
        "prevention": "Improve air circulation around plants.",
        "severity": "Medium"
    },
    "Sooty Mould": {
        "emoji": "🖤",
        "description": "Black fungal growth usually due to sap-sucking insects.",
        "symptoms": "Black, sooty coating on leaves that rubs off.",
        "treatment": "Control honeydew-producing insects. Wash leaves with mild soap solution.",
        "prevention": "Manage aphids, scales, and other sap-sucking pests.",
        "severity": "Low-Medium"
    }
}

CLASSES = list(DISEASE_INFO.keys())

# Custom CSS with animations and advanced styling
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    * {
        font-family: 'Poppins', sans-serif;
        transition: all 0.3s ease;
    }

    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #f8fbf9 0%, #e6f3ec 100%);
        padding: 0 2rem;
    }

    /* Header styling with animation */
    .header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e6b45 0%, #2e8b57 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        margin-bottom: 1.5rem;
        animation: fadeIn 1.2s ease-in-out;
        text-align: center;
        padding: 1rem 0;
        position: relative;
        letter-spacing: -0.5px;
    }

    .header:after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 150px;
        height: 4px;
        background: linear-gradient(90deg, #2e8b57 0%, #3cb371 100%);
        border-radius: 2px;
    }

    /* Subheader styling */
    .subheader {
        font-size: 1.6rem;
        font-weight: 600;
        color: #2a6840;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.7rem;
        position: relative;
    }

    .subheader:after {
        content: "";
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 100px;
        height: 3px;
        background: linear-gradient(90deg, #2e8b57 0%, #3cb371 100%);
        border-radius: 2px;
    }

    /* Card styling with hover effects */
    .disease-card {
        border-radius: 15px;
        padding: 1.8rem;
        margin-bottom: 1.8rem;
        background: white;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        transition: all 0.4s ease;
        border-left: 6px solid #2e8b57;
        position: relative;
        overflow: hidden;
    }

    .disease-card:before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 5px;
        background: linear-gradient(90deg, #2e8b57 0%, #3cb371 100%);
        transform: translateY(-100%);
        transition: transform 0.4s ease;
    }

    .disease-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.15);
    }

    .disease-card:hover:before {
        transform: translateY(0);
    }

    /* Confidence indicators */
    .confidence-high {
        color: #1e6b45;
        font-weight: 700;
        background: rgba(46, 139, 87, 0.15);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
    }

    .confidence-medium {
        color: #d97c00;
        font-weight: 700;
        background: rgba(255, 165, 0, 0.15);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
    }

    .confidence-low {
        color: #cc3300;
        font-weight: 700;
        background: rgba(255, 69, 0, 0.15);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
    }

    /* Progress bar customization */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #2e8b57 0%, #3cb371 100%);
        border-radius: 6px;
        height: 12px;
    }

    /* Button styling */
    .stButton>button {
        border: none;
        background: linear-gradient(90deg, #1e6b45 0%, #2e8b57 100%);
        color: white;
        padding: 0.7rem 1.8rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(30, 107, 69, 0.3);
        position: relative;
        overflow: hidden;
    }

    .stButton>button:before {
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: all 0.6s;
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(30, 107, 69, 0.5);
    }

    .stButton>button:hover:before {
        left: 100%;
    }

    /* File uploader styling */
    .stFileUploader>div>div>div>div {
        border: 2px dashed #2e8b57;
        border-radius: 15px;
        background: rgba(46, 139, 87, 0.05);
        padding: 2rem 1rem;
    }

    /* Animation for results */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }

    .result-animation {
        animation: fadeIn 0.8s ease-out;
    }

    .pulse {
        animation: pulse 2s infinite;
    }

    /* Severity badges */
    .severity-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff3333 100%);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 3px 10px rgba(255, 107, 107, 0.3);
    }

    .severity-medium {
        background: linear-gradient(135deg, #ffd166 0%, #ffb347 100%);
        color: #333;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 3px 10px rgba(255, 209, 102, 0.3);
    }

    .severity-low {
        background: linear-gradient(135deg, #06d6a0 0%, #05a181 100%);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 3px 10px rgba(6, 214, 160, 0.3);
    }

    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #2e8b57;
        cursor: help;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 220px;
        background-color: #1e6b45;
        color: #fff;
        text-align: center;
        border-radius: 10px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -110px;
        opacity: 0;
        transition: opacity 0.3s;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        font-size: 0.9rem;
        font-weight: 400;
    }

    .tooltip .tooltiptext::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #1e6b45 transparent transparent transparent;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    /* Image preview styling */
    .image-preview {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        transition: all 0.4s ease;
        border: 3px solid #fff;
    }

    .image-preview:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fbf9 0%, #e6f3ec 100%);
        box-shadow: 5px 0 15px rgba(0,0,0,0.05);
    }

    .sidebar-header {
        text-align: center;
        padding: 1.5rem 0;
        border-bottom: 1px solid rgba(46, 139, 87, 0.2);
        margin-bottom: 1.5rem;
    }

    .sidebar-header h2 {
        font-size: 1.8rem;
        color: #1e6b45;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }

    .sidebar-header p {
        color: #4a8d65;
        font-size: 0.9rem;
    }

    .disease-ref {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }

    .disease-ref:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.08);
    }

    /* Footer styling */
    .footer {
        text-align: center;
        padding: 1.5rem 0;
        margin-top: 2rem;
        border-top: 1px solid rgba(46, 139, 87, 0.2);
        color: #4a8d65;
        font-size: 0.9rem;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .header {
            font-size: 2.2rem;
        }

        .subheader {
            font-size: 1.4rem;
        }

        .disease-card {
            padding: 1.2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# JavaScript for additional interactivity
st.markdown("""
<script>
    // Add smooth scrolling to all links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // Add animation when elements come into view
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate');
            }
        });
    }, {threshold: 0.1});

    document.querySelectorAll('.animate-on-scroll').forEach(el => {
        observer.observe(el);
    });

    // Add pulse animation to important elements
    setInterval(() => {
        const pulseElements = document.querySelectorAll('.pulse');
        pulseElements.forEach(el => {
            el.classList.remove('pulse');
            void el.offsetWidth; // Trigger reflow
            el.classList.add('pulse');
        });
    }, 4000);
</script>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(checkpoint_path="trained_models/leaf"):
    """Load the trained model"""
    model = resnet34()
    model.fc = nn.Linear(in_features=512, out_features=8)
    checkpoint = torch.load(os.path.join(checkpoint_path, "best.pt"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def preprocess_image(image, image_size=224):
    """Preprocess the uploaded image"""
    image = np.array(image)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (image_size, image_size))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image / 255.
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))[None, :, :, :]
    image = torch.from_numpy(image).float()
    return image


def predict(image, model, device):
    """Make prediction on the image"""
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        output = model(image.to(device))
        prob = softmax(output)
        predicted_prob, predicted_class = torch.max(prob, dim=1)
    return predicted_prob[0], predicted_class[0]


def severity_badge(severity):
    """Return styled severity badge"""
    severity = severity.lower()
    if "high" in severity:
        return f'<span class="severity-high">{severity.upper()}</span>'
    elif "medium" in severity:
        return f'<span class="severity-medium">{severity.upper()}</span>'
    else:
        return f'<span class="severity-low">{severity.upper()}</span>'


def image_to_base64(image):
    """Convert PIL image to base64"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def generate_report(disease_name, confidence, disease_info):
    """Generate a text report for download"""
    return f"""
    🍃 MANGO LEAF DISEASE ANALYSIS REPORT 🍃
    ======================================

    Diagnosis: {disease_name} ({confidence:.2f}% confidence)
    Severity: {disease_info['severity']}

    DESCRIPTION:
    {disease_info['description']}

    SYMPTOMS:
    {disease_info['symptoms']}

    RECOMMENDED TREATMENT:
    {disease_info['treatment']}

    PREVENTION METHODS:
    {disease_info['prevention']}

    Generated by MangoLeaf AI
    {time.strftime("%Y-%m-%d %H:%M:%S")}
    """


def main():
    # Custom header with HTML
    st.markdown("""
    <div class="header pulse">
        <span style="vertical-align: middle;">🍃</span> MangoLeaf AI Disease Classifier
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="font-size: 1.2rem; color: #4a8d65; margin-bottom: 2rem; text-align: center;">
        Leveraging deep learning to accurately identify and diagnose mango leaf diseases. 
        Upload an image below to get instant analysis and treatment recommendations.
    </p>
    """, unsafe_allow_html=True)

    # Sidebar with enhanced features
    with st.sidebar:
        # Logo and title
        st.markdown("""
        <div class="sidebar-header">
            <h2>🍃 MangoLeaf AI</h2>
            <p>Advanced Disease Detection System</p>
        </div>
        """, unsafe_allow_html=True)

        # Navigation menu
        st.markdown("### Navigation")
        menu = ["Home", "How It Works", "Disease Library", "About", "Contact"]
        choice = st.selectbox("", menu, label_visibility="collapsed")

        # Upload section in sidebar
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a mango leaf image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a mango leaf for analysis",
            label_visibility="collapsed"
        )

        # Disease quick reference
        with st.expander("📚 Disease Quick Reference", expanded=True):
            for disease in CLASSES:
                emoji = DISEASE_INFO[disease]["emoji"]
                st.markdown(f"""
                <div class="disease-ref">
                    <div style="margin-bottom: 0.3rem; font-size: 1rem; display: flex; align-items: center;">
                        <span style="font-size: 1.4rem; margin-right: 10px;">{emoji}</span>
                        <strong>{disease}</strong>
                    </div>
                    <div style="font-size: 0.9rem; color: #555;">
                        {DISEASE_INFO[disease]["description"][:55]}...
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Contact information
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.9rem; color: #4a8d65; padding: 1rem 0;">
            <p style="margin-bottom: 0.8rem; font-weight: 600;">Need help? Contact our support team</p>
            <div style="display: flex; justify-content: center; gap: 1rem; margin-bottom: 1rem;">
                <div style="background: white; border-radius: 10px; padding: 0.7rem; box-shadow: 0 3px 10px rgba(0,0,0,0.05);">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">✉️</div>
                    <div>support@mangoleaf.ai</div>
                </div>
                <div style="background: white; border-radius: 10px; padding: 0.7rem; box-shadow: 0 3px 10px rgba(0,0,0,0.05);">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📞</div>
                    <div>+1 (555) 123-4567</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Main content columns
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="subheader">📤 Image Upload & Preview</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            # Display uploaded image with fancy border
            image = Image.open(uploaded_file)
            st.markdown(f"""
            <div class="image-preview">
                <img src="data:image/png;base64,{image_to_base64(image)}" style="width: 100%;"/>
            </div>
            """, unsafe_allow_html=True)

            # Save to temp file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                image.save(tmp_file.name)
                image_path = tmp_file.name
        else:
            image_path = None
            # Placeholder with animation
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f0f9f4 0%, #e1f3e9 100%); 
                        border: 2px dashed #2e8b57; border-radius: 15px; 
                        height: 350px; display: flex; justify-content: center; align-items: center;
                        margin-bottom: 1.5rem; box-shadow: 0 8px 20px rgba(0,0,0,0.05);">
                <div style="text-align: center; color: #4a8d65;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" 
                         fill="none" stroke="#2e8b57" stroke-width="2" stroke-linecap="round" 
                         stroke-linejoin="round" style="margin-bottom: 1.5rem;">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="17 8 12 3 7 8"></polyline>
                        <line x1="12" y1="3" x2="12" y2="15"></line>
                    </svg>
                    <h3 style="color: #1e6b45; margin-bottom: 0.5rem;">Upload a Mango Leaf Image</h3>
                    <p>Supported formats: JPG, JPEG, PNG</p>
                    <p style="margin-top: 1rem; color: #5d9c74; font-size: 0.9rem;">
                        <span class="tooltip">Tips for best results
                            <span class="tooltiptext">Use a clear, well-lit image of a single leaf against a plain background</span>
                        </span>
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="subheader">🔍 Analysis Results</div>', unsafe_allow_html=True)

        if image_path:
            # Loading animation
            with st.spinner("Analyzing the leaf image..."):
                # Simulate processing time for better UX
                time.sleep(1.5)

                # Load model and make prediction
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = load_model().to(device)
                processed_image = preprocess_image(image)
                confidence, class_idx = predict(processed_image, model, device)
                confidence_percent = confidence * 100
                disease_name = CLASSES[class_idx]

                # Get all predictions for the radar chart
                with torch.no_grad():
                    output = model(processed_image.to(device))
                    softmax = nn.Softmax(dim=1)
                    all_probs = softmax(output)[0].cpu().numpy()

            # Results container with animation class
            emoji = DISEASE_INFO[disease_name]["emoji"]
            description = DISEASE_INFO[disease_name]["description"]
            symptoms = DISEASE_INFO[disease_name]["symptoms"]
            confidence_color = "confidence-high" if confidence_percent > 85 else (
                "confidence-medium" if confidence_percent > 60 else "confidence-low")

            st.markdown(f"""
            <div class="result-animation" style="background: white; border-radius: 15px; padding: 1.5rem; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                <h3 style="color: #1e6b45;">{emoji} {disease_name}</h3>
                <p><strong style="color: #1e6b45;">📝 Description:</strong> {description}</p>
                <p><strong style="color: #1e6b45;">{emoji} Symptoms:</strong> {symptoms}</p>
                <p><strong style="color: #1e6b45;">Confidence:</strong> <span class="{confidence_color}">{confidence_percent:.2f}%</span></p>
            </div>
            """, unsafe_allow_html=True)
            # Treatment card with tabs
            tab1, tab2 = st.tabs(["💊 Treatment Recommendations", "🛡️ Prevention Strategies"])

            with tab1:
                st.markdown(f"""
                <div class="disease-card">
                    <h4 style="margin-top: 0; color: #1e6b45; border-bottom: 1px solid #eee; padding-bottom: 0.8rem;">Treatment Recommendations</h4>
                    <p style="font-size: 1.05rem; line-height: 1.6;">{DISEASE_INFO[disease_name]["treatment"]}</p>
                </div>
                """, unsafe_allow_html=True)

            with tab2:
                st.markdown(f"""
                <div class="disease-card">
                    <h4 style="margin-top: 0; color: #1e6b45; border-bottom: 1px solid #eee; padding-bottom: 0.8rem;">Prevention Strategies</h4>
                    <p style="font-size: 1.05rem; line-height: 1.6;">{DISEASE_INFO[disease_name]["prevention"]}</p>
                </div>
                """, unsafe_allow_html=True)

            # All predictions radar chart (simplified)
            st.markdown("### Disease Probability Distribution")
            st.bar_chart({cls: float(prob * 100) for cls, prob in zip(CLASSES, all_probs)})

            # Download report button
            st.download_button(
                label="📥 Download Full Analysis Report",
                data=generate_report(disease_name, confidence_percent, DISEASE_INFO[disease_name]),
                file_name=f"mango_leaf_report_{disease_name.lower().replace(' ', '_')}.txt",
                mime="text/plain"
            )
        else:
            # Results placeholder
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f0f9f4 0%, #e1f3e9 100%); 
                        border-radius: 15px; padding: 2.5rem; text-align: center; color: #4a8d65;
                        box-shadow: 0 8px 20px rgba(0,0,0,0.05); height: 500px; display: flex; 
                        flex-direction: column; justify-content: center;">
                <svg xmlns="http://www.w3.org/2000/svg" width="72" height="72" viewBox="0 0 24 24" 
                     fill="none" stroke="#2e8b57" stroke-width="2" stroke-linecap="round" 
                     stroke-linejoin="round" style="margin-bottom: 1.5rem;">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
                <h3 style="color: #1e6b45; margin-bottom: 0.8rem; font-size: 1.8rem;">Ready to Analyze</h3>
                <p style="font-size: 1.1rem; max-width: 400px; margin: 0 auto 1.5rem;">
                    Upload an image of a mango leaf to get detailed analysis and recommendations
                </p>
                <div style="display: inline-block; background: white; padding: 0.8rem 1.5rem; 
                            border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); 
                            color: #2e8b57; font-weight: 600;">
                    <span class="pulse">⬅️ Upload an image using the sidebar</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        © 2023 MangoLeaf AI | Advanced Plant Disease Detection System
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()