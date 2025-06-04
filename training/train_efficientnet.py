import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import shutil # Ensure shutil is imported at the top level

# Config
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transforms - same for both datasets
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def get_next_version_path(base_output_dir, model_subdir_name):
    """
    Finds the next available versioned path for a model subdirectory.
    Example: if 'base_output_dir/model_subdir_name' exists,
    it tries 'base_output_dir/model_subdir_name_v2', then _v3, etc.
    Returns the full path for the new version.
    """
    target_path = os.path.join(base_output_dir, model_subdir_name)
    if not os.path.exists(target_path):
        return target_path

    version = 2
    while True:
        versioned_name = f"{model_subdir_name}_v{version}"
        versioned_path = os.path.join(base_output_dir, versioned_name)
        if not os.path.exists(versioned_path):
            return versioned_path
        version += 1

def train_model(data_dir, model_save_dir, model_save_name):
    print(f"\nStarting training for dataset: {data_dir}")

    # Load dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load pretrained EfficientNet-B0
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    model.train()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # Save model
    os.makedirs(model_save_dir, exist_ok=True)
    save_path = os.path.join(model_save_dir, model_save_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

def main():
    base_preprocessing_dir = "preprocessing"
    models_output_dir = "C:\\\\Users\\\\Owner\\\\Desktop\\\\preprocessing\\\\models"

    # Train on 3_canny_edges
    canny_data_dir = os.path.join(base_preprocessing_dir, "no_paper", "3_canny_edges")
    canny_data_dir_with = os.path.join(base_preprocessing_dir, "with_paper", "3_canny_edges")
    canny_dataset_dir = "temp_canny_dataset"

    # Prepare combined dataset folder structure for canny edges (ImageFolder requires one root)
    # We'll symlink or copy the images from both classes into canny_dataset_dir/{class}/ folders

    # import shutil # shutil is already imported at the top

    def prepare_dataset(src_dirs, dst_dir):
        # src_dirs: dict class->path, dst_dir: root
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        for cls, path in src_dirs.items():
            dst_cls_dir = os.path.join(dst_dir, cls)
            os.makedirs(dst_cls_dir, exist_ok=True)
            for fname in os.listdir(path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    src_file = os.path.join(path, fname)
                    dst_file = os.path.join(dst_cls_dir, fname)
                    # Copy file
                    shutil.copy2(src_file, dst_file)

    # Prepare canny dataset
    prepare_dataset({'no_paper': canny_data_dir, 'with_paper': canny_data_dir_with}, canny_dataset_dir)
    train_model(canny_dataset_dir, "models/3_canny_edges", "efficientnet_b0_canny.pth")

    # Train on 4_dilated_edges
    dilated_data_dir = os.path.join(base_preprocessing_dir, "no_paper", "4_dilated_edges")
    dilated_data_dir_with = os.path.join(base_preprocessing_dir, "with_paper", "4_dilated_edges")
    dilated_dataset_dir = "temp_dilated_dataset"

    # Prepare dilated dataset
    prepare_dataset({'no_paper': dilated_data_dir, 'with_paper': dilated_data_dir_with}, dilated_dataset_dir)
    train_model(dilated_dataset_dir, "models/4_dilated_edges", "efficientnet_b0_dilated.pth")

    # Cleanup temp folders (optional)
    if os.path.exists(canny_dataset_dir):
        shutil.rmtree(canny_dataset_dir)
    if os.path.exists(dilated_dataset_dir):
        shutil.rmtree(dilated_dataset_dir)

    # Ensure the base models_output_dir exists
    os.makedirs(models_output_dir, exist_ok=True)

    # move the models to the specified models_output_dir with versioning
    src_canny_path = "models/3_canny_edges"
    if os.path.exists(src_canny_path):
        dest_canny_base_name = os.path.basename(src_canny_path)
        versioned_dest_canny_path = get_next_version_path(models_output_dir, dest_canny_base_name)
        shutil.move(src_canny_path, versioned_dest_canny_path)
        print(f"Moved {src_canny_path} to {versioned_dest_canny_path}")
    else:
        print(f"Source path {src_canny_path} does not exist. Skipping move.")

    src_dilated_path = "models/4_dilated_edges"
    if os.path.exists(src_dilated_path):
        dest_dilated_base_name = os.path.basename(src_dilated_path)
        versioned_dest_dilated_path = get_next_version_path(models_output_dir, dest_dilated_base_name)
        shutil.move(src_dilated_path, versioned_dest_dilated_path)
        print(f"Moved {src_dilated_path} to {versioned_dest_dilated_path}")
    else:
        print(f"Source path {src_dilated_path} does not exist. Skipping move.")

if __name__ == "__main__":
    main()
