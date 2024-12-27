import torch
from torchvision.transforms import functional as F
from PIL import Image
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

DEFAULT_MODEL_PATH = "MangaTranslatorApp/trained_faster_rcnn.pth"

def load_model(model_path=DEFAULT_MODEL_PATH, num_classes=3):
    """
    Yüklenmiş Faster R-CNN modelini hazırlar.

    Args:
        model_path (str): Kaydedilmiş modelin dosya yolu.
        num_classes (int): Sınıf sayısı.

    Returns:
        model: Yüklenmiş model.
    """

    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

def predict(image_path, device=torch.device('cpu')):
    """
    Görsel üzerinde tahmin yapar.

    Args:
        image_path (str): Tahmin yapılacak görselin dosya yolu.
        model: Yüklü model.
        device (torch.device): Tahmin için kullanılacak cihaz.

    Returns:
        dict: {'boxes': list}
    """
    model=load_model()
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

    output = outputs[0] 
    boxes = output['boxes'].cpu().numpy()

    return {'boxes': boxes.tolist()}

