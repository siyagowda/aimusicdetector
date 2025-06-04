from pathlib import Path
import torch
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import torch.nn.functional as F

def get_model(weight_path):
    # Load the pre-trained ResNet-18 model
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

    # Modify the last layer of the model
    num_classes = 2 # number of classes in dataset
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(weight_path))
    return model

def predict():

    mel_dir = Path("/vol/bitbucket/sg2121/fyp/aimusicdetector/inference/features/Mel_Spectrogram")
    mel_paths = list(mel_dir.glob("*.png"))  

    mfcc_dir = Path("/vol/bitbucket/sg2121/fyp/aimusicdetector/inference/features/MFCC")
    mfcc_paths = list(mfcc_dir.glob("*.png"))  

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    predictions = []

    weights = {'mel_cnn': 0.5, 'mfcc_cnnt': 0.5}

    model = get_model("/vol/bitbucket/sg2121/fyp/aimusicdetector/music_cnn/large/mel-spec/cur_model.pt")
    model.eval()

    for mel_path in mel_paths:
        # Load image and apply transform
        image = Image.open(mel_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(image_tensor)
            prob = F.softmax(output, dim=1)  # Get probabilities
            predictions.append(prob.cpu() * weights["mel_cnn"])

    model = get_model("/vol/bitbucket/sg2121/fyp/aimusicdetector/music_cnn/large/mfcc/cur_model.pt")
    model.eval()

    for mfcc_path in mfcc_paths:
        # Load image and apply transform
        image = Image.open(mfcc_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(image_tensor)
            prob = F.softmax(output, dim=1)  # Get probabilities
            predictions.append(prob.cpu() * weights["mfcc_cnnt"])

    print(len(predictions))
    # Average probabilities across all chunks
    avg_prob = torch.mean(torch.cat(predictions, dim=0), dim=0)
    predicted_label = torch.argmax(avg_prob).item()
    confidence = avg_prob[predicted_label].item()

    label_name = "AI-generated" if predicted_label == 0 else "Human-generated"
    print(f"\nPrediction: {label_name} (Confidence: {confidence:.2f})")

    return label_name, confidence