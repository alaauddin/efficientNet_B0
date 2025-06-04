import torch
from torchvision import transforms, models
from PIL import Image
import os
import torch.nn as nn

# Load the model
def load_model(model_path):
    # Recreate the model architecture
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # 2 classes
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Predict function
def predict_image(image_path, model):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prob, predicted = torch.max(probabilities, 1)

    
    return predicted.item(), prob.item()

def run_inference(model_path, test_dir):
    model = load_model(model_path)
    
    for label_dir in ['test_no_paper', 'test_with_paper']:
        folder_path = os.path.join(test_dir, label_dir)
        print(f"\n📂 Predicting images in: {folder_path}")
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, img_file)
                pred_class, prob = predict_image(image_path, model)
                print(f"🖼️ {img_file} → Predicted Class: {pred_class} (Probability: {prob:.4f})")

# Example usage
if __name__ == "__main__":
    # Choose one of the models
    model_path = "C://Users//Owner//Desktop//preprocessing//models//4_dilated_edges_v6//efficientnet_b0_dilated.pth"  # or "models/4_dilated_edges/model.pt"
    test_dir = "C://Users//Owner//Desktop//preprocessing//inference"
    run_inference(model_path, test_dir)
