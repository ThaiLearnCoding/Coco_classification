import torch
import clip
from tqdm import tqdm
from sklearn.metrics import classification_report

def evaluate_zero_shot(model, dataloader, class_names, device):
    model.clip_model.eval()
    all_preds = []
    all_labels = []
    
    # Tạo text prompts: "A photo of a [class]"
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(device)
    
    with torch.no_grad():
        text_features = model.clip_model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        for images, labels in tqdm(dataloader, desc="Zero-shot Eval"):
            images = images.to(device)
            image_features = model.clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Tính độ tương đồng
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            preds = similarity.argmax(dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    return classification_report(all_labels, all_preds, target_names=class_names)

def train_few_shot(model, train_loader, criterion, optimizer, epochs, device):
    model.classifier.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")