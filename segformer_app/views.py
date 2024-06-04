# segformer_app/views.py
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from .forms import ImageUploadForm
from .segformer_model import load_model
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

model = load_model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def upload_image(request):
    print("upload_image view called")  # Debugging line
    if request.method == 'POST':
        print("POST request received")  # Debugging line
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            print("Form is valid")  # Debugging line
            image = form.cleaned_data['image']
            fs = FileSystemStorage()
            filename = fs.save(image.name, image)
            uploaded_file_url = fs.url(filename)
            print(f"Image saved at {uploaded_file_url}")  # Debugging line

            # Process the uploaded image with the model
            img = Image.open(fs.path(filename)).convert("RGB")
            original_size = img.size  # Get the original image size (width, height)
            transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
            ])
            input_image = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_image).logits
                outputs = torch.nn.functional.interpolate(outputs, size=(1024, 1024), mode='bilinear', align_corners=False)
                prediction = outputs.argmax(dim=1).squeeze().cpu().numpy()

            # Convert prediction to displayable format
            display_prediction = np.zeros_like(prediction, dtype=np.uint8)
            display_prediction[prediction == 0] = 0  # Skin
            display_prediction[prediction == 1] = 255  # Lesion

            # Calculate cNF Intensity (lesion/skin) ratio as a percentage
            lesion_area = np.sum(prediction == 1)
            skin_area = np.sum(prediction == 0)
            cNF_intensity = (lesion_area / (skin_area + lesion_area)) * 100 if skin_area > 0 else float('inf')
            cNF_intensity = round(cNF_intensity, 1)

            # Resize the prediction to the original image size
            prediction_resized = Image.fromarray(display_prediction).resize(original_size, Image.NEAREST)
            prediction_image_path = 'prediction_' + filename
            prediction_resized.save(fs.path(prediction_image_path))

            print(f"Redirecting to result page with URLs: {uploaded_file_url}, {fs.url(prediction_image_path)}")  # Debugging line
            return render(request, 'segformer_app/result.html', {
                'original': uploaded_file_url,
                'prediction': fs.url(prediction_image_path),
                'cNF_intensity': cNF_intensity,
            })
        else:
            print("Form is not valid")  # Debugging line
    else:
        print("GET request received, rendering upload form")  # Debugging line
        form = ImageUploadForm()
    return render(request, 'segformer_app/upload.html', {'form': form})
