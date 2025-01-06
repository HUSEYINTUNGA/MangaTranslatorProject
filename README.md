# Manga Translator Project

## Projenin Amacı

Manga Translator Project, manga ve çizgi roman görsellerindeki metinleri tespit edip kullanıcının tercih ettiği bir dile çevirmeyi hedefleyen bir sistemdir. Bu proje, dil bariyerlerini aşıp manga deneyimini daha erişilebilir hale getirmek amacıyla geliştirilmiştir.

---

## Özellikler

### 1. Metin Tespiti

Faster R-CNN modeli, görsellerdeki metin kutularını tespit etmek için kullanılmıştır. Bu aşamada:

- Görsel normalize edilerek modele giriş olarak verilir.
- Metin kutularının koordinatları belirlenir.

### 2. Optik Karakter Tanıma (OCR)

EasyOCR kullanılarak tespit edilen metin kutularındaki içerikler dijital metne dönüştürülür. Bu aşamada:

- Her bir metin kutusu, Faster R-CNN tarafından tespit edilen koordinatlar kullanılarak görselden kesilir (cropping).
- EasyOCR, kesilen kutulardaki metinleri dijital formata dönüştürür. Bu işlem, metinlerin daha yüksek doğrulukla okunmasını sağlar çünkü OCR işlemi doğrudan ilgili bölge üzerinde çalışır ve gürültüyü azaltır.

### 3. Çeviri

Google Translate API ile metinler hedef dile çevrilir. Bu aşamada:

- Dil algılama yapılır.
- Kullanıcının seçtiği dile göre çeviri sağlanır.

### 4. Görsel İşleme

Çevrilen metinler, orijinal görselin üzerine yerleştirilir. Bu aşamada:

- Tespit edilen kutular beyaz bir arka planla doldurulur.
- Çevrilen metin uygun bir tasarımla kutulara yazılır.

---

## Model Eğitimi

### Veri Seti

- Kaynak: Kaggle’dan sağlanan [Manga Text Detection Dataset](https://www.kaggle.com/datasets/naufalahnaf17/manga-text-detection).
- Etiketleme: RoboFlow kullanılarak COCO formatında etiketleme yapıldı.
- Toplam Görsel Sayısı: 494 (Eğitim: 349, Doğrulama: 100, Test: 45).

### Eğitim Süreci

- **Model**: ResNet50 tabanlı Faster R-CNN.
- **Hiperparametreler**: Öğrenme oranı: 0.005, Momentum: 0.9, Epoch sayısı: 20.
- **Performans Metrikleri**:
  - Precision: 0.7649
  - Recall: 0.9334
  - F1 Skoru: 0.8338
  - Doğruluk: 0.7358

---

## Teknik Detaylar

### Kullanılan Teknolojiler

- **Backend**: Django
- **Frontend**: HTML, CSS
- **Model**: Faster R-CNN (PyTorch)
- **OCR**: EasyOCR
- **Çeviri**: Google Translate API

### Sistem Mimarisi

1. Kullanıcı bir görsel yükler ve çeviri dilini seçer.
2. Django backend’i aşağıdaki adımları gerçekleştirir:
   - Faster R-CNN modeli ile metin kutuları tespit edilir.
   - Tespit edilen kutular görsel üzerinden kesilir ve OCR işlemi yalnızca bu kesilen kutular üzerinde gerçekleştirilir. Bu yöntem, OCR işleminin doğruluğunu artırır ve gereksiz alanların işlenmesini önler.
   - OCR kullanılarak metin dijital formata dönüştürülür.
   - Metinler Google Translate API ile çevrilir.
   - Çevrilen metinler görsellere yerleştirilir.
3. Kullanıcıya işlenmiş görsel sunulur.

---
## Performans Değerlendirme

### Metrikler

- **IoU (Intersection over Union)**: 0.7 eşik değeri ile bounding box tespiti.
- **Precision**: 0.7649
- **Recall**: 0.9334
- **F1 Skoru**: 0.8338
- **Accuracy**: 0.7358

### Gözlemler

- Model, karmaşık arka planlarda performans kaybı yaşayabilir.
- Beyaz arka plan uygulaması, metinlerin daha okunaklı hale gelmesini sağlamıştır.
- Google Translate API’nin çeviri kalitesi bağlam gerektiren metinlerde sınırlı kalabilir.
- Küçük fontlu veya el yazısı tarzındaki metinler, OCR tarafından doğru şekilde okunamamış ve bu durum çeviri sürecini olumsuz etkilemiştir.
- Düşük kontrastlı görseller veya yoğun görsel gürültü içeren arka planlar, hem metin tespitinde hem de OCR işlemlerinde zorluklara neden olmuştur.

---

## Kurulum ve Kullanım

### Gereksinimler

- Python 3.8+
- PyTorch
- EasyOCR
- Django
- Google Translate API anahtarı

### Kurulum

1. Depoyu klonlayın:
   ```bash
   git clone https://github.com/HUSEYINTUNGA/manga-translator.git
   cd manga-translator
   ```
2. Gereksinimleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```
3. Google Translate API anahtarını yapılandırın.

### Çalıştırma

1. Django sunucusunu başlatın:
   ```bash
   python manage.py runserver
   ```
2. Tarayıcınızda [http://localhost:8000](http://localhost:8000) adresine giderek uygulamayı kullanın.

---

## Katkılar

Katkı sağlamak için bir çekme isteği (pull request) oluşturun veya sorunları bildirin.

---

