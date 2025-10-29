import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

# 1. Preprocessing & Transform

def preprocess_simple(img):
    img = img.convert('RGB')
    img = img.resize((224, 224))
    return img

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])


# 2. Grad-CAM

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        output = self.model(input_tensor)
        self.model.zero_grad()
        # Binary classification
        if output.shape[1] == 1:
            target_class = target_class if target_class is not None else 1
            loss = output[0,0] if target_class == 1 else -output[0,0]
        else:
            target_class = target_class if target_class is not None else output.argmax(dim=1).item()
            loss = output[0,target_class]
        loss.backward()
        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224,224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# 3. Prediction + Grad-CAM

def predict_and_show_gradcam(model, image, device, show_heatmap=True):
    processed_img = preprocess_simple(image)
    tensor_img = transform(processed_img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(tensor_img)
        prob = torch.sigmoid(output).item()
        pred = 1 if prob >= 0.5 else 0

    classes = {0: 'Normal', 1: 'Pneumonia'}
    warning = " (LOW CONFIDENCE)" if 0.4 <= prob <= 0.6 else ""

    if show_heatmap:
        target_layer = model.layer4[1].conv3
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate(tensor_img, target_class=pred)

        # Invert CAM for better visualization
        cam_inverted = 1 - cam
        heatmap = cv2.applyColorMap(np.uint8(255*cam_inverted), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap)/255
        original_img = np.array(processed_img)/255.0
        overlay = heatmap*0.4 + original_img*0.6

        original_img = np.clip(original_img, 0, 1)
        heatmap = np.clip(heatmap, 0, 1)
        overlay = np.clip(overlay, 0, 1)

        from PIL import Image
        original_pil = Image.fromarray((original_img*255).astype(np.uint8))
        heatmap_pil = Image.fromarray((heatmap*255).astype(np.uint8))
        overlay_pil = Image.fromarray((overlay*255).astype(np.uint8))

        col1, col2, col3 = st.columns(3)
        col1.image(original_pil, use_container_width=True)
        col1.markdown("**Original**")
        col2.image(heatmap_pil, use_container_width=True)
        col2.markdown("**Grad-CAM Heatmap**")
        col3.image(overlay_pil, use_container_width=True)
        col3.markdown(f"**{classes[pred]}{warning}**")

    else:
        st.image(processed_img, caption="Original", use_container_width=True)

   
    # Display Prediction & Explanation
  
    if pred == 1:  # Pneumonia
      confidence = prob * 100
      st.write(f"**Prediction:** Pneumonia ({confidence:.2f}%) {warning}")
    else:  # Normal
      confidence = (1 - prob) * 100
      st.write(f"**Prediction:** Normal ({confidence:.2f}%) {warning}")


    # Explanation text for user
    if pred == 1:  # Pneumonia
        explanation = (
            "Pneumonia X-ray detected.\n\n"
            "The model is confident .\n"
            "Grad-CAM highlights local lung regions with opacity or patchy infiltrates, "
            "consistent with pneumonia lesions.\n"
            "Bright yellow/red spots on the overlay indicate affected areas."
        )
    else:  # Normal
        explanation = (
            "Normal X-ray detected.\n\n"
            "The model is confident .\n"
            "Grad-CAM highlights diffuse or structural regions, such as ribs, diaphragm, or broad lung areas.\n"
            "No localized pneumonia patterns are detected."
        )

    # Display explanation text
    st.markdown(f'<p class="explanation-text">{explanation}</p>', unsafe_allow_html=True)
   



# 4. Load Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(weights=ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 1)
model.load_state_dict(torch.load("best_resnet50_normal.pth", map_location=device))
model = model.to(device)
model.eval()


# 5. Streamlit UI

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state.page = "welcome"

if st.session_state.page == "welcome":
    local_css("style.css")
    st.markdown('<h1 class="welcome-title">Welcome to Pneumonia Detection </h1>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-subtitle">Enhanced Pneumonia Detection from Chest X-ray Images Using ResNet50 and Gradient-based Visual Explanations (Grad-CAM).</p>', unsafe_allow_html=True)

    st.markdown('<div class="center-button">', unsafe_allow_html=True)
    if st.button("Double Click to Start"):
        st.session_state.page = "detection"
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "detection":
    local_css("style.css")
    
    # Wrap everything in the detection-background div
    st.markdown('<div class="detection-background">', unsafe_allow_html=True)
    
    # Detection page title
    st.markdown('<h1 class="detection-title">Pneumonia Detection with Grad-CAM</h1>', unsafe_allow_html=True)

    # Checkbox and uploader
    show_heatmap = st.checkbox("Show Grad-CAM", value=True)
    uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["png","jpg","jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        predict_and_show_gradcam(model, image, device, show_heatmap=show_heatmap)

    # Close the div
    st.markdown('</div>', unsafe_allow_html=True)