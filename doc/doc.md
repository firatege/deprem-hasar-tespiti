Deprem Hasar Tahminleme ve Analiz
Projesi (Python & PyTorch)
1. Proje Özeti
Bu proje, uydu görüntüleri (Optical & SAR) ve derin öğrenme tekniklerini kullanarak deprem
sonrası bina hasarını otonom bir şekilde tespit etmeyi ve sismik verilerle hasar olasılığını
tahminlemeyi amaçlar. Proje, karmaşık backend yapılarından arındırılmış, tamamen veri bilimi
ve modelleme odaklı bir Python iş akışı üzerine kuruludur.
2. Veri Kaynakları ve Hazırlık
Modelin eğitimi ve test süreci için iki ana veri kaynağı kullanılacaktır:
● xBD Dataset: Deprem, sel ve yangın gibi afetlerin öncesi ve sonrası yüksek çözünürlüklü
uydu görüntülerini ve bina bazlı hasar etiketlerini (No Damage, Minor, Major, Destroyed)
içerir.
● Google Earth Engine (GEE): Gerçek zamanlı analizler için Sentinel-1 (Radar) ve
Sentinel-2 (Optik) verilerinin çekilmesinde kullanılacaktır.
Veri Ön İşleme (Preprocessing)
Uydu görüntüleri çok büyük boyutlu olduğu için modelin işleyebileceği "Tile" (karo) sistemine
dönüştürülmelidir:
● Görüntüler 512x512 veya 1024x1024 piksellik parçalara bölünür.
● Data Augmentation: Modelin genelleme yeteneğini artırmak için döndürme, aynalama ve
renk manipülasyonları uygulanır.
3. Model Mimarisi: Siamese U-Net
İki farklı zaman dilimindeki (deprem öncesi ve sonrası) değişimi yakalamak için Siamese (İkiz)
ağ yapısı tercih edilmiştir.
Bileşen Teknoloji / Yöntem Görev
Encoder (Özellik Çıkarıcı) ResNet-34 / EfficientNet Öncesi ve sonrası
görüntülerden mekansal
özellikleri (features) çıkarır.
Siamese Fusion Feature Differencing İki görüntü arasındaki fark
haritasını (feature map
subtraction) hesaplar.
Decoder (Segmentasyon) U-Net Decoder Fark haritasını kullanarak
hasarlı bina maskelerini
(segmentation mask) üretir.

4. Eğitim Stratejisi

Modelin eğitiminde karşılaşılan en büyük sorun, "sağlam bina" sayısının "yıkık bina" sayısından
çok daha fazla olmasıdır (class imbalance).
Kayıp Fonksiyonları (Loss Functions)
Bu dengesizliği gidermek için hibrit bir loss yapısı kullanılır:
● Focal Loss: Modelin yanlış tahmin edilen zor örneklere daha fazla odaklanmasını sağlar.
● Dice Loss: Tahmin edilen hasar maskesi ile gerçek maske arasındaki örtüşmeyi
(Intersection over Union) maksimize eder.
5. Tahminleme (Inference) ve Çıktılar
Eğitilen model, sadece görüntü işlemekle kalmayıp şu tahminleri üretebilir:
1. Hasar Segmentasyonu: Harita üzerinde hangi binaların yıkıldığının piksel bazlı
gösterimi.
2. Olasılık Haritası: Sismik veriler (PGA) ve bina öznitelikleriyle birleşerek, henüz görüntü
alınmamış alanlar için hasar olasılığı tahmini.
6. Proje Yol Haritası
● Hafta 1-2: GEE Python API entegrasyonu ve xBD datasetinin indirilip "patchify" edilmesi.
● Hafta 3-4: PyTorch ile Siamese U-Net mimarisinin kodlanması ve Transfer Learning ile
eğitimin başlatılması.
● Hafta 5: Modelin validation seti (örneğin Japonya veya Fas depremi) üzerinde test
edilmesi.
● Hafta 6: Sonuçların Python (Matplotlib/Plotly) veya basit bir Streamlit arayüzü ile
görselleştirilmesi.
Not: Bu proje, ağır backend mimarileri yerine doğrudan model performansı ve coğrafi veri
analitiğine odaklanmaktadır.
Referanslar
1. xView2 Challenge: https://www.xview2.org/
2. Segmentation Models PyTorch: GitHub link