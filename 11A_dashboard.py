"""
OSTEOPOROSIS DETECTION DASHBOARD - FINAL WORKING VERSION
=========================================================
Properly loads models saved directly from training
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

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
</style>
""", unsafe_allow_html=True)

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
    """Load all available trained models"""
    models_dict = {}
    class_names = ['Normal', 'Osteopenia', 'Osteoporosis']
    
    # Sydney's ResNet18 Model
    if os.path.exists('sydney_best_model.pth'):
        try:
            sydney_model = models.resnet18(weights=None)  # No pretrained weights
            num_ftrs = sydney_model.fc.in_features
            sydney_model.fc = nn.Linear(num_ftrs, 3)
            
            # Load the state dict
            sydney_model.load_state_dict(torch.load('sydney_best_model.pth', map_location=device, weights_only=True))
            sydney_model.to(device)
            sydney_model.eval()
            models_dict['Sydney (ResNet18)'] = sydney_model
            st.success("✅ Sydney's ResNet18 loaded")
        except Exception as e:
            st.error(f"❌ Sydney's model failed: {str(e)[:200]}")
    
    # ResNet50 Model  
    if os.path.exists('resnet50_best_model.pth'):
        try:
            resnet50_model = models.resnet50(weights=None)
            num_ftrs = resnet50_model.fc.in_features
            resnet50_model.fc = nn.Linear(num_ftrs, 3)
            
            resnet50_model.load_state_dict(torch.load('resnet50_best_model.pth', map_location=device, weights_only=True))
            resnet50_model.to(device)
            resnet50_model.eval()
            models_dict['Michael (ResNet50)'] = resnet50_model
            st.success("✅ Michael's ResNet50 loaded")
        except Exception as e:
            st.error(f"❌ Michael's model failed: {str(e)[:200]}")
    
    # Dia's EfficientNet-B3 (if available)
    if os.path.exists('best_osteoporosis_model.pth'):
        try:
            dia_model = models.efficientnet_b3(weights=None)
            in_features = 1536
            dia_model.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 3)
            )
            dia_model.load_state_dict(torch.load('best_osteoporosis_model.pth', map_location=device, weights_only=True))
            dia_model.to(device)
            dia_model.eval()
            models_dict['Dia (EfficientNet-B3)'] = dia_model
            st.success("✅ Dia's EfficientNet-B3 loaded")
        except Exception as e:
            st.warning(f"⚠️ Dia's model not loaded (different architecture expected)")
    
    # Michael's model (if different from ResNet50)
    if os.path.exists('michael_model.pth'):
        try:
            michael_model = models.resnet50(weights=None)
            num_ftrs = michael_model.fc.in_features
            michael_model.fc = nn.Linear(num_ftrs, 3)
            
            michael_model.load_state_dict(torch.load('michael_model.pth', map_location=device, weights_only=True))
            michael_model.to(device)
            michael_model.eval()
            models_dict['Michael (ResNet50)'] = michael_model
            st.success("✅ Michael's ResNet50 loaded")
        except Exception as e:
            st.error(f"❌ Michael's model failed: {str(e)[:200]}")
    
    if not models_dict:
        st.error("❌ No models could be loaded! Check your .pth files.")
        return None, class_names
    
    return models_dict, class_names

# ========== PREDICTION FUNCTIONS ==========
def predict_single_model(model, image_tensor, device):
    """Get prediction from a single model"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        return probs.cpu().numpy()[0]

def ensemble_predict(models_dict, image, device, class_names):
    """Ensemble prediction using all available models"""
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
        st.error("Please ensure your .pth model files are in the same directory as this script.")
        st.stop()
    
    st.success(f"✅ Successfully loaded {len(models_dict)} model(s)")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.write(f"""
        This application uses **{len(models_dict)} deep learning model(s)** 
        to analyze knee X-rays and detect osteoporosis.
        
        **Models Loaded:**
        """)
        for model_name in models_dict.keys():
            st.write(f"- 🔷 {model_name}")
        
        st.write("""
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
                    
                    # Store in session state
                    st.session_state['analyzed'] = True
                    st.session_state['prediction'] = prediction
                    st.session_state['confidence'] = confidence
                    st.session_state['all_predictions'] = all_predictions
                    st.session_state['ensemble_probs'] = ensemble_probs
                    
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
        if st.session_state.get('analyzed', False):
            prediction = st.session_state['prediction']
            confidence = st.session_state['confidence']
            all_predictions = st.session_state['all_predictions']
            ensemble_probs = st.session_state['ensemble_probs']
            
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
                            st.progress(float(probs[i]), text=f"{class_name}: {probs[i]*100:.1f}%")
            
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
                
                **Confidence Level:** {confidence:.1f}%
                - 90-100%: Very High Confidence
                - 75-90%: High Confidence
                - 50-75%: Moderate Confidence
                - Below 50%: Low Confidence (manual review recommended)
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

# ========== RUN APP ==========
if __name__ == "__main__":
    # Initialize session state
    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = False
    
    main()