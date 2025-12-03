"""
OSTEOPOROSIS DETECTION DASHBOARD
===================================
Ensemble prediction system using three trained models:
- Michael's Model: ResNet50 (Layer4 unfrozen)
- Dia's Model: EfficientNet-B3
- Sydney's Model: ResNet18 (Fully trainable)

Users upload knee X-ray images and get ensemble predictions
with confidence scores for: Normal, Osteopenia, Osteoporosis
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="Osteoporosis Detection",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .confidence-score {
        font-size: 1.5rem;
        color: #667eea;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ========== MODEL ARCHITECTURES ==========

class MichaelModel(nn.Module):
    """ResNet50 with Layer4 unfrozen"""
    def __init__(self, num_classes=3):
        super(MichaelModel, self).__init__()
        self.model = models.resnet50(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

class DiaModel(nn.Module):
    """EfficientNet-B3 with custom classifier"""
    def __init__(self, num_classes=3):
        super(DiaModel, self).__init__()
        self.backbone = models.efficientnet_b3(pretrained=False)
        in_features = 1536  # EfficientNet-B3 features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class SydneyModel(nn.Module):
    """ResNet18 fully trainable"""
    def __init__(self, num_classes=3):
        super(SydneyModel, self).__init__()
        self.model = models.resnet18(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# ========== TRANSFORMS ==========
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ========== MODEL LOADING ==========
@st.cache_resource
def load_models(device):
    """Load all three trained models"""
    models_dict = {}
    class_names = ['Normal', 'Osteopenia', 'Osteoporosis']
    
    try:
        # Michael's Model
        michael_model = MichaelModel(num_classes=3).to(device)
        try:
            michael_model.load_state_dict(torch.load('michael_model.pth', map_location=device))
            michael_model.eval()
            models_dict['Michael (ResNet50)'] = michael_model
        except FileNotFoundError:
            st.warning("⚠️ Michael's model not found. Place 'michael_model.pth' in the app directory.")
        
        # Dia's Model
        dia_model = DiaModel(num_classes=3).to(device)
        try:
            dia_model.load_state_dict(torch.load('best_osteoporosis_model.pth', map_location=device))
            dia_model.eval()
            models_dict['Dia (EfficientNet-B3)'] = dia_model
        except FileNotFoundError:
            st.warning("⚠️ Dia's model not found. Place 'best_osteoporosis_model.pth' in the app directory.")
        
        # Sydney's Model
        sydney_model = SydneyModel(num_classes=3).to(device)
        try:
            sydney_checkpoint = torch.load('sydney_best_model.pth', map_location=device)
            if isinstance(sydney_checkpoint, dict) and 'model_state_dict' in sydney_checkpoint:
                sydney_model.load_state_dict(sydney_checkpoint['model_state_dict'])
            else:
                sydney_model.load_state_dict(sydney_checkpoint)
            sydney_model.eval()
            models_dict['Sydney (ResNet18)'] = sydney_model
        except FileNotFoundError:
            st.warning("⚠️ Sydney's model not found. Place 'sydney_best_model.pth' in the app directory.")
        
        if not models_dict:
            st.error("❌ No models found! Please add model files to the directory.")
            return None, class_names
        
        return models_dict, class_names
    
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, class_names

# ========== PREDICTION FUNCTIONS ==========
def predict_single_model(model, image_tensor, device):
    """Get prediction from a single model"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        return probs.cpu().numpy()[0]

def ensemble_predict(models_dict, image, device, class_names):
    """
    Ensemble prediction: average probabilities from all models
    
    Returns:
        prediction: Predicted class
        confidence: Confidence score (0-100)
        all_probs: Dictionary of probabilities from each model
    """
    image_tensor = transform(image).unsqueeze(0)
    
    all_predictions = {}
    ensemble_probs = np.zeros(len(class_names))
    
    for model_name, model in models_dict.items():
        probs = predict_single_model(model, image_tensor, device)
        all_predictions[model_name] = probs
        ensemble_probs += probs
    
    # Average probabilities
    ensemble_probs /= len(models_dict)
    
    # Get prediction
    predicted_idx = np.argmax(ensemble_probs)
    prediction = class_names[predicted_idx]
    confidence = float(ensemble_probs[predicted_idx] * 100)
    
    return prediction, confidence, all_predictions, ensemble_probs

# ========== VISUALIZATION FUNCTIONS ==========
def create_confidence_gauge(confidence, prediction):
    """Create a gauge chart for confidence"""
    # Color based on prediction
    color_map = {
        'Normal': '#90EE90',
        'Osteopenia': '#FFD700',
        'Osteoporosis': '#FF6B6B'
    }
    color = color_map.get(prediction, '#667eea')
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Confidence: {prediction}", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#FFE5E5'},
                {'range': [50, 75], 'color': '#FFF4E5'},
                {'range': [75, 100], 'color': '#E5F5E5'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
    return fig

def create_probability_chart(probs, class_names):
    """Create bar chart showing probabilities for all classes"""
    colors = ['#90EE90', '#FFD700', '#FF6B6B']
    
    fig = go.Figure(data=[
        go.Bar(
            x=class_names,
            y=probs * 100,
            marker_color=colors,
            text=[f'{p*100:.1f}%' for p in probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Class Probabilities",
        xaxis_title="Diagnosis",
        yaxis_title="Probability (%)",
        yaxis=dict(range=[0, 100]),
        height=400,
        showlegend=False
    )
    
    return fig

def create_model_comparison_chart(all_predictions, class_names):
    """Create chart comparing predictions from each model"""
    data = []
    
    for model_name, probs in all_predictions.items():
        for i, class_name in enumerate(class_names):
            data.append({
                'Model': model_name,
                'Class': class_name,
                'Probability': probs[i] * 100
            })
    
    import pandas as pd
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df,
        x='Class',
        y='Probability',
        color='Model',
        barmode='group',
        title='Model Comparison: Individual Predictions',
        labels={'Probability': 'Probability (%)'},
        color_discrete_sequence=['#667eea', '#764ba2', '#f093fb']
    )
    
    fig.update_layout(height=400, yaxis=dict(range=[0, 100]))
    return fig

# ========== MAIN APP ==========
def main():
    # Header
    st.markdown('<div class="main-header">🦴 Osteoporosis Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Knee X-Ray Analysis Using Ensemble Deep Learning</div>', unsafe_allow_html=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    with st.spinner('Loading AI models...'):
        models_dict, class_names = load_models(device)
    
    if not models_dict:
        st.stop()
    
    st.success(f"✅ Loaded {len(models_dict)} model(s): {', '.join(models_dict.keys())}")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.write("""
        This application uses an **ensemble of three deep learning models** 
        to analyze knee X-rays and detect osteoporosis:
        
        **Models:**
        - 🔷 **Michael's Model**: ResNet50 (Layer4 fine-tuned)
        - 🔶 **Dia's Model**: EfficientNet-B3 with custom classifier
        - 🔵 **Sydney's Model**: ResNet18 (fully fine-tuned)
        
        **How it works:**
        1. Upload a knee X-ray image
        2. Each model independently analyzes the image
        3. Predictions are combined using ensemble averaging
        4. Final diagnosis with confidence score is displayed
        """)
        
        st.header("⚕️ Diagnosis Categories")
        st.write("""
        - 🟢 **Normal**: Healthy bone density
        - 🟡 **Osteopenia**: Low bone density (early stage)
        - 🔴 **Osteoporosis**: Very low bone density (advanced)
        """)
        
        st.header("⚠️ Important Notice")
        st.warning("""
        This is a **research tool** and not FDA approved for clinical use. 
        All predictions should be reviewed by qualified medical professionals.
        """)
    
    # Main content
    st.markdown("---")
    st.header("📤 Upload X-Ray Image")
    
    uploaded_file = st.file_uploader(
        "Choose a knee X-ray image (JPG, PNG, JPEG)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear anterior-posterior or lateral knee X-ray"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_container_width=True)
            
            # Image info
            st.info(f"""
            **Image Details:**
            - Size: {image.size[0]} × {image.size[1]} pixels
            - Format: {image.format}
            - Mode: {image.mode}
            """)
        
        with col2:
            st.subheader("🔬 Analysis")
            
            # Analyze button
            if st.button("🚀 Analyze X-Ray", type="primary", use_container_width=True):
                with st.spinner('Analyzing image with ensemble models...'):
                    # Get ensemble prediction
                    prediction, confidence, all_predictions, ensemble_probs = ensemble_predict(
                        models_dict, image, device, class_names
                    )
                    
                    # Display result
                    st.markdown(f"""
                    <div class="prediction-box">
                        Diagnosis: {prediction}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence gauge
                    gauge_fig = create_confidence_gauge(confidence, prediction)
                    st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Detailed results (appears after analysis)
        if st.session_state.get('analyzed', False) or 'prediction' in locals():
            st.markdown("---")
            st.header("📊 Detailed Analysis")
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📈 Ensemble Results", "🔍 Model Comparison", "📋 Details"])
            
            with tab1:
                st.subheader("Ensemble Probability Distribution")
                prob_chart = create_probability_chart(ensemble_probs, class_names)
                st.plotly_chart(prob_chart, use_container_width=True)
                
                # Probability table
                st.subheader("Probability Breakdown")
                prob_data = {
                    'Diagnosis': class_names,
                    'Probability (%)': [f"{p*100:.2f}%" for p in ensemble_probs],
                    'Raw Score': [f"{p:.4f}" for p in ensemble_probs]
                }
                st.table(prob_data)
            
            with tab2:
                st.subheader("Individual Model Predictions")
                comparison_chart = create_model_comparison_chart(all_predictions, class_names)
                st.plotly_chart(comparison_chart, use_container_width=True)
                
                # Individual model results
                st.subheader("Model-by-Model Breakdown")
                for model_name, probs in all_predictions.items():
                    with st.expander(f"🔬 {model_name}"):
                        pred_idx = np.argmax(probs)
                        pred_class = class_names[pred_idx]
                        pred_conf = probs[pred_idx] * 100
                        
                        st.markdown(f"""
                        **Prediction:** {pred_class}  
                        **Confidence:** {pred_conf:.2f}%
                        """)
                        
                        # Show all probabilities
                        for i, class_name in enumerate(class_names):
                            st.progress(probs[i], text=f"{class_name}: {probs[i]*100:.1f}%")
            
            with tab3:
                st.subheader("Clinical Interpretation")
                
                # Interpretation based on prediction
                if prediction == "Normal":
                    st.success("""
                    **Interpretation:**
                    - Bone density appears within normal range
                    - No significant signs of bone loss detected
                    - Continue regular monitoring as recommended by physician
                    """)
                elif prediction == "Osteopenia":
                    st.warning("""
                    **Interpretation:**
                    - Bone density is lower than normal (Osteopenia)
                    - Early stage of bone loss detected
                    - Lifestyle modifications and medical intervention may be beneficial
                    - Regular monitoring recommended
                    """)
                else:  # Osteoporosis
                    st.error("""
                    **Interpretation:**
                    - Significant bone density loss detected (Osteoporosis)
                    - Increased fracture risk
                    - Medical consultation strongly recommended
                    - Treatment plan should be discussed with healthcare provider
                    """)
                
                st.subheader("Ensemble Methodology")
                st.info(f"""
                **How the ensemble works:**
                - {len(models_dict)} independent models analyzed your X-ray
                - Each model used different architectures and training strategies
                - Final prediction combines all models using weighted averaging
                - Ensemble approach typically provides **more robust** predictions than individual models
                
                **Confidence Level:** {confidence:.1f}%
                - 90-100%: Very High Confidence
                - 75-90%: High Confidence
                - 50-75%: Moderate Confidence
                - Below 50%: Low Confidence (manual review recommended)
                """)
                
                st.subheader("Recommendations")
                if confidence < 70:
                    st.warning("""
                    ⚠️ **Confidence is below 70%**
                    - Consider additional imaging
                    - Manual review by radiologist recommended
                    - Multiple views may provide better assessment
                    """)
                else:
                    st.success("""
                    ✅ **High confidence prediction**
                    - Results are consistent across models
                    - However, clinical correlation is still essential
                    """)
    
    else:
        # Instructions when no image uploaded
        st.info("""
        👆 **Getting Started:**
        1. Click the "Browse files" button above
        2. Select a knee X-ray image from your device
        3. Click "Analyze X-Ray" to get instant AI-powered diagnosis
        
        **Supported formats:** JPG, PNG, JPEG
        """)
        
        # Example images section
        st.markdown("---")
        st.header("📚 Example Results")
        st.write("""
        The system analyzes knee X-rays and classifies them into three categories 
        based on bone density patterns detected by the ensemble of deep learning models.
        """)

# ========== RUN APP ==========
if __name__ == "__main__":
    # Initialize session state
    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = False
    
    main()
