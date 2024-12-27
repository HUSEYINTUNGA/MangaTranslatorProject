# Projenin Bölümleri

## 1. Metin Tespiti
Faster R-CNN modeli kullanılarak manga görsellerindeki metin kutuları tespit edilir.
Model, metinlerin konumlarını belirlemek için eğitilmiş bir sinir ağıdır. Bu aşamada:
  -Görselin giriş olarak alınması,
  -Metin kutularının koordinatlarının tespit edilmesi sağlanır.

## 2. Optik Karakter Tanıma (OCR)
Tespit edilen metin kutularındaki içerikler EasyOCR kütüphanesi kullanılarak okunur. OCR aşaması şu işlemleri içerir:
  -Her bir metin kutusunun kesilmesi (cropping),
  -İçeriğin dijital formata dönüştürülmesi.

## 3. Çeviri
OCR ile elde edilen metinler Google Translate API kullanılarak hedef dile çevrilir. Çeviri aşamasında:
  -Metinlerin dil algılaması yapılır,
  -Kullanıcının seçtiği dile göre çeviri gerçekleştirilir.

## 4. Görsel İşleme
Çevrilen metinler orijinal görselin üzerine eklenir. Görsel işleme süreci şu işlemleri kapsar:
  -Tespit edilen kutuların beyaz bir arka planla doldurulması,
  -Çevrilen metnin kutulara yazdırılması.

# Model Eğitimi

## Veri Seti
Model, Kaggle'dan ([http://](https://www.kaggle.com/datasets/naufalahnaf17/manga-text-detection) adrsinden alınan görseller "Hüseyin TUNGA" tarafından etiketlenmiştir.
Model, bu manga görselleri üzerinde eğitildi. Veri seti şu özelliklere sahiptir:
  -Etiket Formatı: COCO JSON formatı,
  -İçerik: Manga görsellerindeki metin kutuları ve sınıf bilgileri.

## Eğitim Süreci
Model eğitimi için Faster R-CNN (ResNet50 tabanlı) kullanıldı. Eğitim aşamaları şunlardır:
  -Model Özelleştirme: ResNet50 özellik çıkarıcı olarak kullanıldı ve son katmanlar manga metin tespiti için ayarlandı.
  -Hiperparametreler: Öğrenme oranı, momentum gibi parametreler optimize edildi.
  -Eğitim Metrikleri:
    --Doğruluk: %73
    --F1 Skoru: %83

# Teknik Detaylar

## Kullanılan Teknolojiler
  -Backend: Django
  -Frontend: HTML, CSS
  -Model: Faster R-CNN (PyTorch kullanılarak)
  -OCR: EasyOCR
  -Çeviri: Google Translate API

## Sistem Mimarisi
  -Kullanıcı bir görsel yükler ve çeviri dilini seçer.
  -Django backend’i şu işlemleri yürütür:
    --Görseli Faster R-CNN modeli ile işler ve metin kutularını tespit eder.
    --OCR ile metni okur ve çevirir.
    --Çevrilen metni işlenmiş görsel üzerine yerleştirir.
  -İşlenmiş görsel kullanıcıya sunulur.

