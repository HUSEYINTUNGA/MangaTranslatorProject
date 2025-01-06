# Manga Translator Project

## Projenin Amacı
Manga Translator Project, manga ve çizgi roman görsellerindeki metinleri tespit edip çevirerek, kullanıcılara bu içerikleri istedikleri dillerde sunmayı amaçlayan bir uygulamadır. Proje, görüntü işleme, optik karakter tanıma (OCR) ve çeviri teknolojilerini birleştirerek bu işlemleri otomatize eder.

---

## Özellikler

### 1. Metin Tespiti
Faster R-CNN modeli kullanılarak manga görsellerindeki metin kutuları tespit edilir. Bu aşamada:
- Görsel giriş olarak alınır.
- Metin kutularının koordinatları belirlenir.

### 2. Optik Karakter Tanıma (OCR)
Tespit edilen metin kutularındaki içerikler EasyOCR kütüphanesi kullanılarak okunur. Bu aşamada:
- Her bir metin kutusu kesilir (cropping).
- Metin dijital formata dönüştürülür.

### 3. Çeviri
OCR ile elde edilen metinler Google Translate API kullanılarak hedef dile çevrilir. Bu aşamada:
- Metinlerin dili algılanır.
- Kullanıcının seçtiği dile göre çeviri yapılır.

### 4. Görsel İşleme
Çevrilen metinler orijinal görselin üzerine eklenir. Bu aşamada:
- Tespit edilen kutular beyaz bir arka planla doldurulur.
- Çevrilen metin kutulara yazılır.

---

## Model Eğitimi

### Veri Seti
- Kaggle'dan alınan bir veri seti kullanıldı ([Manga Text Detection Dataset](https://www.kaggle.com/datasets/naufalahnaf17/manga-text-detection)).
- Görseller "Hüseyin TUNGA" tarafından etiketlenmiştir.
- Etiket formatı: COCO JSON.
- Veri seti, manga görsellerindeki metin kutuları ve sınıf bilgilerini içerir.

### Eğitim Süreci
- **Model**: ResNet50 tabanlı Faster R-CNN.
- **Model Özelleştirme**: ResNet50 özellik çıkarıcı olarak kullanıldı ve son katmanlar manga metin tespiti için ayarlandı.
- **Hiperparametreler**: Öğrenme oranı ve momentum optimize edildi.
- **Eğitim Metrikleri**:
  - Doğruluk: %73
  - F1 Skoru: %83

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
2. Django backend’i aşağıdaki işlemleri gerçekleştirir:
   - Görseli Faster R-CNN modeli ile işler ve metin kutularını tespit eder.
   - OCR ile metni okur ve çevirir.
   - Çevrilen metni işlenmiş görsel üzerine yerleştirir.
3. Kullanıcıya işlenmiş görsel sunulur.

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
2. Gerekli Python kütüphanelerini yükleyin:
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

## Geliştirici Notları
- OCR ve Faster R-CNN modellerini optimize ederek çalışma hızı arttırılabilir.
- Alternatif çeviri API'leri entegre edilerek çeviri kalitesi geliştirilebilir.
- Kullanıcı dostu bir arayüz eklenebilir.

---

## Katkılar
Katkı sağlamak için bir çekme isteği (pull request) oluşturun veya sorunları bildirin.

