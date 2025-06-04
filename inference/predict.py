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

def get_predictions(dir, model_path, weight, predictions):
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    preds = predictions

    model = get_model(model_path)
    model.eval()

    paths = list(dir.glob("*.png")) 

    for path in paths:
        # Load image and apply transform
        image = Image.open(path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(image_tensor)
            prob = F.softmax(output, dim=1)  # Get probabilities
            preds.append(prob.cpu() * weight)

    return preds

def predict():
    predictions = []

    weights = {'mel_cnn': 0.5, 'mfcc_cnnt': 0.5}

    # Mel-Spectrogram CNN
    predictions = get_predictions(Path("/vol/bitbucket/sg2121/fyp/aimusicdetector/inference/features/Mel-Spectrogram"),
                    "/vol/bitbucket/sg2121/fyp/aimusicdetector/music_cnn/large/mel-spec/cur_model.pt",
                    weights["mel_cnn"], predictions)
    
    # MFCC CNN-T
    predictions = get_predictions(Path("/vol/bitbucket/sg2121/fyp/aimusicdetector/inference/features/MFCC"),
                    "/vol/bitbucket/sg2121/fyp/aimusicdetector/music_cnn/large/mfcc/cur_model.pt",
                    weights["mfcc_cnnt"], predictions)

    # Average probabilities across all chunks
    avg_prob = torch.mean(torch.cat(predictions, dim=0), dim=0)
    predicted_label = torch.argmax(avg_prob).item()
    confidence = avg_prob[predicted_label].item()

    label_name = "AI-generated" if predicted_label == 0 else "Human-generated"
    print(f"\nPrediction: {label_name} (Confidence: {confidence:.2f})")

    return label_name, confidence