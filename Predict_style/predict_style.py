import onnxruntime as ort
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import time
from codecarbon import EmissionsTracker


def predict_image_with_onnx(image_pil, onnx_model_path):
    # Classes cibles (à adapter si nécessaire)
    class_names = ["Ghibli", "Jojo", "Arcane", "Simpsons", "Autre"]
   
    # Début du chronomètre global
    start_global = time.time()


    # Charger le modèle ONNX
    #ort_session = ort.InferenceSession(onnx_model_path)
    
    # Transformation standard
    transform = transforms.Compose([
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
     ])

    # Version transformation de l'entraînement avec de moins bon résultats
#     transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
# ])



    # Charger et transformer l'image
    image = image_pil.convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    input_array = input_tensor.numpy().astype(np.float32)


    # Suivi des émissions
    tracker = EmissionsTracker(
        measure_power_secs=1,
        log_level='ERROR',
        project_name="ConvNeXt_Training"
    )
    tracker._country_iso_code = "FRA"
    tracker._cloud_provider = "GCP"
    tracker._cloud_region = "eu-west-3"


    tracker.start()
    start_inference = time.time()

    ort_session = ort.InferenceSession(onnx_model_path)
    # Inférence
    ort_outputs = ort_session.run(None, {'input': input_array})


    end_inference = time.time()
    emissions = tracker.stop()


    # Résultats
    probs_tensor = torch.softmax(torch.tensor(ort_outputs[0]), dim=1)
    pred_idx = torch.argmax(probs_tensor, dim=1).item()
    predicted_class = class_names[pred_idx]
    predicted_prob = probs_tensor[0][pred_idx].item() * 100
    probabilities = [prob.item() * 100 for prob in probs_tensor[0]]
    
    if predicted_prob < 50:
        predicted_class = "Autre"
    else:
        predicted_class = class_names[pred_idx]

    end_global = time.time()


    return {
        "classe_predite": predicted_class,
        "probabilite_predite": predicted_prob,
        "temps_inference_s": end_inference - start_inference,
        "emissions_CO2_kg": emissions,
        "temps_total_s": end_global - start_global,
        "probabilites_par_classe": dict(zip(class_names, probabilities))
    }



# if __name__ == "__main__" : 
    # results = predict_image_with_onnx(
    #     # image_path=r"/Users/nivo/Desktop/Hackhaton/AIthicalMinds-/test_simpsons.jpg",
    #     image_pil = Image.open("AIthicalMinds-/Predict_style/test_simpsons.jpg"),
    #     onnx_model_path="best_model.onnx"
    # )

#     print("\nRésultats de l'inférence:")
#     print(f"Classe prédite: {results['classe_predite']} ({results['probabilite_predite']:.2f}%)")


#     print("\nProbabilités par classe:")
#     for cls, prob in results['probabilites_par_classe'].items():
#         print(f"{cls}: {prob:.2f}%")


#     print("\nMétriques de performance:")
#     print(f"Temps total du test : {results['temps_total_s']:.4f} s")
#     print(f"Temps d'inférence : {results['temps_inference_s']:.4f} s")
#     print(f"Émissions CO₂: {results['emissions_CO2_kg']:.6f} kg")












