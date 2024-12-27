import os
import torch
import torchvision
import numpy as np
from tqdm import tqdm                                              
import seaborn as sns                
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

def calculate_iou(box1, box2):
    """
    İki bounding box arasındaki IoU'yu hesaplar.

    Parametreler:
    - box1: [xmin, ymin, xmax, ymax] formatında birinci box.
    - box2: [xmin, ymin, xmax, ymax] formatında ikinci box.

    Döndürür:
    - IoU değeri (0 ile 1 arasında).
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def plot_confusion_matrix(tp, fp, fn, save_path=None):
    """
    Confusion matrix'i çiz ve isteğe bağlı olarak dosyaya kaydet.

    Parametreler:
    - tp: Doğru pozitiflerin sayısı.
    - fp: Yanlış pozitiflerin sayısı.
    - fn: Yanlış negatiflerin sayısı.
    - save_path: Matrisin kaydedileceği dosya yolu (varsayılan: None).

    Döndürmez.
    """
    cm = np.array([[tp, fp], [fn, 0]])
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def calculate_accuracy(tp, fp, fn):
    """
    Doğruluk oranını hesaplar.

    Parametreler:
    - tp: Doğru pozitiflerin sayısı.
    - fp: Yanlış pozitiflerin sayısı.
    - fn: Yanlış negatiflerin sayısı.

    Döndürür:
    - Accuracy değeri.
    """
    total = tp + fp + fn
    return tp / total if total > 0 else 0

class MangaDataset(CocoDetection):
    """
    COCO veri setini Manga için özelleştirilmiş şekilde işler.

    Parametreler:
    - root: Veri setinin kök klasörü.
    - anno: Anotasyon dosyasının yolu.
    - transform: Görüntüye uygulanacak dönüşümler (varsayılan: None).
    """
    def __init__(self, root, anno, transform=None):
        super(MangaDataset, self).__init__(root, anno)
        self.transform = transform

    def __getitem__(self, index):
        image, target = super(MangaDataset, self).__getitem__(index)
        if self.transform is not None:
            image = self.transform(image)

        boxes = []
        labels = []

        for obj in target:
            xmin, ymin, width, height = obj['bbox']
            if width > 0 and height > 0:
                boxes.append([xmin, ymin, xmin + width, ymin + height])
                labels.append(obj['category_id'])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64)
        }

        return image, target

def transform(image):
    """
    Görüntüyü tensöre dönüştürür.

    Parametreler:
    - image: PIL görüntü nesnesi.

    Döndürür:
    - Tensor formatında görüntü.
    """
    return F.to_tensor(image)

def collate_fn(batch):
    """
    Batch işleme için veri kümelerini gruplayan fonksiyon.

    Parametreler:
    - batch: İşlenecek batch verisi.

    Döndürür:
    - Tuple formatında batch verisi.
    """
    return tuple(zip(*batch))

def get_model(num_classes):
    """
    Faster R-CNN modelini döndürür.

    Parametreler:
    - num_classes: Sınıf sayısı.

    Döndürür:
    - Faster R-CNN modeli.
    """
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

def evaluate_predictions(outputs, targets, iou_threshold=0.7):
    """
    Tahminler ile gerçek etiketler arasındaki metrikleri hesaplar.

    Parametreler:
    - outputs: Model tahminleri.
    - targets: Gerçek etiketler.
    - iou_threshold: IoU eşik değeri.

    Döndürür:
    - Precision, Recall, F1 Skoru değerleri ve TP, FP, FN sayıları.
    """
    tp, fp, fn = 0, 0, 0

    for output, target in zip(outputs, targets):
        pred_boxes = output['boxes'].cpu().numpy()
        true_boxes = target['boxes'].cpu().numpy()

        matched = []
        for true_box in true_boxes:
            ious = [calculate_iou(true_box, pred_box) for pred_box in pred_boxes]
            max_iou = max(ious) if ious else 0

            if max_iou > iou_threshold:
                tp += 1
                matched.append(pred_boxes[ious.index(max_iou)])
            else:
                fn += 1

        for pred_box in pred_boxes:
            if not any(np.array_equal(pred_box, matched_box) for matched_box in matched):
                fp += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1, tp, fp, fn

def train_and_validate(model, train_loader, valid_loader, optimizer, lr_scheduler, device, num_epochs=20):
    """
    Modeli eğitir ve doğrulama işlemini yapar.

    Parametreler:
    - model: Eğitim yapılacak model.
    - train_loader: Eğitim veri yükleyicisi.
    - valid_loader: Doğrulama veri yükleyicisi.
    - optimizer: Optimizasyon algoritması.
    - lr_scheduler: Öğrenme oranı planlayıcısı.
    - device: Eğitim cihazı (CPU veya GPU).
    - num_classes: Sınıf sayısı.(30 epochs).
    - num_epochs: Epoch sayısı.

    Döndürür:
    - TP, FP, FN toplamları.
    """
    output_dir = "MangaTranslatorApp/Model_Metrics/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metrics_file_path = os.path.join(output_dir, "epoch_metrics.txt")

    final_tp, final_fp, final_fn = 0, 0, 0

    with open(metrics_file_path, "w") as metrics_file:
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            total_batches = 0
            for images, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                epoch_loss += losses.item()
                total_batches += 1

            average_loss = epoch_loss / total_batches
            lr_scheduler.step()

            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {average_loss:.4f}")

            model.eval()
            all_precisions, all_recalls, all_f1s, all_accuracies = [], [], [], []
            tp_sum, fp_sum, fn_sum = 0, 0, 0

            with torch.no_grad():
                for images, targets in tqdm(valid_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}"):
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    outputs = model(images)
                    precision, recall, f1, tp, fp, fn = evaluate_predictions(outputs, targets)
                    accuracy = calculate_accuracy(tp, fp, fn)

                    all_precisions.append(precision)
                    all_recalls.append(recall)
                    all_f1s.append(f1)
                    all_accuracies.append(accuracy)

                    tp_sum += tp
                    fp_sum += fp
                    fn_sum += fn

            final_tp += tp_sum
            final_fp += fp_sum
            final_fn += fn_sum

            mean_precision = sum(all_precisions) / len(all_precisions)
            mean_recall = sum(all_recalls) / len(all_recalls)
            mean_f1 = sum(all_f1s) / len(all_f1s)
            mean_accuracy = sum(all_accuracies) / len(all_accuracies)

            metrics_file.write(f"Epoch {epoch + 1}:\n")
            metrics_file.write(f"Precision: {mean_precision:.4f}\n")
            metrics_file.write(f"Recall: {mean_recall:.4f}\n")
            metrics_file.write(f"F1 Score: {mean_f1:.4f}\n")
            metrics_file.write(f"Accuracy: {mean_accuracy:.4f}\n")
            metrics_file.write(f"Average Loss: {average_loss:.4f}\n\n")

            print(f"Epoch {epoch + 1}: Precision = {mean_precision:.4f}, Recall = {mean_recall:.4f}, F1 Score = {mean_f1:.4f}, Accuracy = {mean_accuracy:.4f}")

    print("Training and validation complete.")
    return final_tp, final_fp, final_fn

def test_model(model, test_loader, device):
    """
    Modeli test eder ve metrikleri hesaplar.

    Parametreler:
    - model: Test edilecek model.
    - test_loader: Test veri yükleyicisi.
    - device: Test cihazı (CPU veya GPU).

    Döndürür:
    - tp: Doğru pozitiflerin toplam sayısı.
    - fp: Yanlış pozitiflerin toplam sayısı.
    - fn: Yanlış negatiflerin toplam sayısı.
    """
    model.eval()
    all_precisions, all_recalls, all_f1s, all_accuracies = [], [], [], []
    tp_sum, fp_sum, fn_sum = 0, 0, 0

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing Model"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            precision, recall, f1, tp, fp, fn = evaluate_predictions(outputs, targets)
            accuracy = calculate_accuracy(tp, fp, fn)

            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)
            all_accuracies.append(accuracy)

            tp_sum += tp
            fp_sum += fp
            fn_sum += fn

    mean_precision = sum(all_precisions) / len(all_precisions)
    mean_recall = sum(all_recalls) / len(all_recalls)
    mean_f1 = sum(all_f1s) / len(all_f1s)
    mean_accuracy = sum(all_accuracies) / len(all_accuracies)

    print(f"Test Results:")
    print(f"Precision = {mean_precision:.4f}, Recall = {mean_recall:.4f}, F1 Score = {mean_f1:.4f}, Accuracy = {mean_accuracy:.4f}")

    return tp_sum, fp_sum, fn_sum

def run_training():
    """
    Eğitim, doğrulama ve test süreçlerini yürütür.

    - Eğitim ve doğrulama aşamasında model optimize edilir ve metrikler hesaplanır.
    - Test aşamasında final metrikler hesaplanır ve raporlanır.
    - Sonuçlar dosyaya kaydedilir.
    """
    dataset_path = "MangaTranslatorApp/TrainAndValidationDataset/"


    train_path = os.path.join(dataset_path, 'train/')
    train_anno = os.path.join(train_path, '_annotations.coco.json')
    train_data = MangaDataset(train_path, train_anno, transform=transform)
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=collate_fn)

    valid_path = os.path.join(dataset_path, 'valid/')
    valid_anno = os.path.join(valid_path, '_annotations.coco.json')
    valid_data = MangaDataset(valid_path, valid_anno, transform=transform)
    valid_loader = DataLoader(valid_data, batch_size=2, shuffle=False, collate_fn=collate_fn)

    test_path = os.path.join(dataset_path, 'test/')
    test_anno = os.path.join(test_path, '_annotations.coco.json')
    test_data = MangaDataset(test_path, test_anno, transform=transform)
    test_loader = DataLoader(test_data, batch_size=2, shuffle=False, collate_fn=collate_fn)


    num_classes = len(train_data.coco.cats) + 1
    model = get_model(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


    print("Starting training and validation...")
    final_tp, final_fp, final_fn = train_and_validate(model, train_loader, valid_loader, optimizer, lr_scheduler, device)

    validation_confusion_matrix_path = "MangaTranslatorApp/Model_Metrics/validation_confusion_matrix.png"
    plot_confusion_matrix(final_tp, final_fp, final_fn, save_path=validation_confusion_matrix_path)
    print(f"Validation Confusion Matrix saved to {validation_confusion_matrix_path}")

    print("Starting test phase...")
    test_tp, test_fp, test_fn = test_model(model, test_loader, device)


    test_confusion_matrix_path = "MangaTranslatorApp/Model_Metrics/test_confusion_matrix.png"
    plot_confusion_matrix(test_tp, test_fp, test_fn, save_path=test_confusion_matrix_path)
    print(f"Test Confusion Matrix saved to {test_confusion_matrix_path}")


    model_save_path = os.path.join(os.path.dirname(__file__), "trained_faster_rcnn.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


