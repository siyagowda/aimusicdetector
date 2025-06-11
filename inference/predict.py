from pathlib import Path
import torch
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertConfig, DistilBertForSequenceClassification, get_scheduler

def get_cnn_model(weight_path):
    # Load the pre-trained ResNet-18 model
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

    # Modify the last layer of the model
    num_classes = 2 # number of classes in dataset
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(weight_path))
    return model

def get_cnn_predictions(dir, model_path, weight, predictions):
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    preds = []

    model = get_cnn_model(model_path)
    model.eval()

    paths = list(dir.glob("*.png")) 

    for path in paths:
        # Load image and apply transform
        image = Image.open(path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(image_tensor)
            prob = F.softmax(output, dim=1)  # Get probabilities
            preds.append(prob.cpu())

    if preds:
        model_avg = torch.mean(torch.cat(preds, dim=0), dim=0)  # shape: [num_classes]
        predictions.append(model_avg * weight)  # Apply weight after averaging

    return predictions

def get_lyrics_model(weight_path):
    config = DistilBertConfig.from_pretrained("distilbert-base-uncased", num_labels=2)
    model = DistilBertForSequenceClassification(config)  
    model.load_state_dict(torch.load(weight_path))
    return model


def get_lyrics_predictions(dir, model_path, weight, predictions):
    preds = []

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = get_lyrics_model(model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    paths = list(dir.glob("*.txt")) 

    for path in paths:
        text = totext(path)

        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            prob = F.softmax(output.logits, dim=1)  # Use .logits for HuggingFace models
            preds.append(prob.cpu())

    if preds:
        model_avg = torch.mean(torch.cat(preds, dim=0), dim=0)  # shape: [num_classes]
        predictions.append(model_avg * weight)  # Apply weight after averaging

    return predictions

class ConvTransformerClassifier(nn.Module):
    def __init__(self, n_classes, d_model=128, nhead=4, num_layers=2, input_shape=(1, 128, 256)):
        super(ConvTransformerClassifier, self).__init__()

        # --- CNN Feature Extractor ---
        conv_layers = []
        in_channels = input_shape[0]
        for _ in range(4):
            conv_layers += [
                nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1),
                nn.BatchNorm2d(d_model),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            ]
            in_channels = d_model

        self.cnn = nn.Sequential(*conv_layers)

        # --- Flatten + Positional Encoding ---
        self.flatten = nn.Flatten(2)  # flatten spatial dims into a sequence
        self.positional_encoding = PositionalEncoding(d_model)

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Classification Head ---
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.bn = nn.BatchNorm1d(d_model)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: (B, 1, 128, 256)
        x = self.cnn(x)  # (B, C, H', W') => (B, 128, 8, 16)
        B, C, H, W = x.shape

        x = self.flatten(x)            # (B, C, H*W)  → sequence
        x = x.permute(0, 2, 1)         # (B, seq_len, C)
        x = self.positional_encoding(x)

        x = self.transformer(x)        # (B, seq_len, C)
        x = x.permute(0, 2, 1)         # (B, C, seq_len)
        x = self.avgpool(x).squeeze(-1)  # (B, C)

        x = self.bn(x)
        x = self.fc(x)                 # (B, n_classes)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

def get_cnnt_model(weight_path):
    
    model = ConvTransformerClassifier(n_classes=2, d_model=128, nhead=4, num_layers=2)
    model.load_state_dict(torch.load(weight_path))

    return model

def get_cnnt_predictions(dir, model_path, weight, predictions):
    
    preds = []

    model = get_cnnt_model(model_path)
    model.eval()

    paths = list(dir.glob("*.pt")) 

    for path in paths:
        # Load image and apply transform
        tensor = torch.load(path) 
        tensor = tensor.unsqueeze(0)  # Shape becomes (1, C, H, W)

        with torch.no_grad():
            output = model(tensor)
            prob = F.softmax(output, dim=1)  # Get probabilities
            preds.append(prob.cpu())

    if preds:
        model_avg = torch.mean(torch.cat(preds, dim=0), dim=0)  # shape: [num_classes]
        predictions.append(model_avg * weight)  # Apply weight after averaging

    return predictions



def predict(fast=False):
    predictions = []

    if fast:
        config = {
            "mel_cnnt": {
                "path": Path("/vol/bitbucket/sg2121/fyp/aimusicdetector/inference/features/Mel_Spectrogram_tensor"),
                "model": "/vol/bitbucket/sg2121/fyp/aimusicdetector/music_transformer/large/mel-spec/model_weights.pt",
                "weight": 1.0,
                "fn": get_cnnt_predictions
            }
        }
    else:
        config = {
            "mel_cnn": {
                "path": Path("/vol/bitbucket/sg2121/fyp/aimusicdetector/inference/features/Mel_Spectrogram"),
                "model": "/vol/bitbucket/sg2121/fyp/aimusicdetector/music_cnn/large/mel-spec/cur_model.pt",
                "weight": 0.25,
                "fn": get_cnn_predictions
            },
            "mfcc_cnn": {
                "path": Path("/vol/bitbucket/sg2121/fyp/aimusicdetector/inference/features/MFCC"),
                "model": "/vol/bitbucket/sg2121/fyp/aimusicdetector/music_cnn/large/mfcc/cur_model.pt",
                "weight": 0.15,
                "fn": get_cnn_predictions
            },
            "cqt_cnn": {
                "path": Path("/vol/bitbucket/sg2121/fyp/aimusicdetector/inference/features/CQT"),
                "model": "/vol/bitbucket/sg2121/fyp/aimusicdetector/music_cnn/large/cqt/cur_model.pt",
                "weight": 0.05,
                "fn": get_cnn_predictions
            },
            "mel_cnnt": {
                "path": Path("/vol/bitbucket/sg2121/fyp/aimusicdetector/inference/features/Mel_Spectrogram_tensor"),
                "model": "/vol/bitbucket/sg2121/fyp/aimusicdetector/music_transformer/large/mel-spec/model_weights.pt",
                "weight": 0.25,
                "fn": get_cnnt_predictions
            },
            "mfcc_cnnt": {
                "path": Path("/vol/bitbucket/sg2121/fyp/aimusicdetector/inference/features/MFCC_tensor"),
                "model": "/vol/bitbucket/sg2121/fyp/aimusicdetector/music_transformer/large/mfcc/model_weights.pt",
                "weight": 0.20,
                "fn": get_cnnt_predictions
            },
            "lyrics": {
                "path": Path("/vol/bitbucket/sg2121/fyp/aimusicdetector/inference/features/temp_chunks"),
                "model": "/vol/bitbucket/sg2121/fyp/aimusicdetector/lyric_detection/large/best_model_randomsearch.pt",
                "weight": 0.10,
                "fn": get_lyrics_predictions
            }
        }

    # Run predictions
    for name, cfg in config.items():
        if cfg["weight"] > 0:
            predictions = cfg["fn"](cfg["path"], cfg["model"], cfg["weight"], predictions)

    # Sum weighted averages from each model
    final_prob = torch.stack(predictions).sum(dim=0)  # shape: [num_classes]
    predicted_label = torch.argmax(final_prob).item()
    confidence = final_prob[predicted_label].item()


    label_name = "AI-generated" if predicted_label == 0 else "Human-generated"
    print(f"\nPrediction: {label_name} (Confidence: {confidence:.2f})")

    return label_name, confidence