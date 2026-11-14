# Sosyal Mühendislik Metinlerinde Duygu ve Aciliyet Analizi ile Oltalama Tespiti

**Görkem Poyrazoğlu** — ORCID: https://orcid.org/0009-0008-8998-7601  
**Begüm Düger** — ORCID: https://orcid.org/0009-0005-4847-172X  
Gazi University, Information Technologies – Security Program, Ankara, Türkiye

## Gerçek deney tamamlandı — 14 Kasım 2026

Yalnızca [naserabdullahalam/phishing-email-dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) veri kümesinin **sürüm 1** içindeki `phishing_email.csv` dosyası kullanıldı. Başka veri veya sentetik araştırma verisi kullanılmadı. Kaggle indirmesi GitHub Actions ile, veri denetimi, eğitim ve test ChatGPT çalışma konteynerinde CPU üzerinde yapıldı.

**Ölçüm dosyaları:** [metrics.csv](results/real_run_20261114/metrics.csv) · [Veri kaynağı, ayrım ve doğrulama kaydı](results/real_run_20261114/verified_results.json).

| Model | Doğruluk % | Kesinlik % | Duyarlılık % | Pozitif F1 % | AP |
|---|---:|---:|---:|---:|---:|
| A — TF-IDF + lojistik regresyon | 99,10 | 99,07 | 99,22 | 99,14 | 0,999238 |
| B — 14 duygu/aciliyet göstergesi + lojistik regresyon | 60,51 | 78,50 | 33,54 | 47,00 | 0,658643 |
| C — TF-IDF + 14 gösterge + lojistik regresyon | 99,08 | 99,06 | 99,18 | 99,12 | 0,999246 |

A modelinde TN/FP/FN/TP = 7766/80/67/8502; B'de 7059/787/5695/2874; C'de 7765/81/70/8499. Her model aynı **16.415 test kaydında** değerlendirildi. Eşik 0,5'tir.

**Ana bulgu:** C−A pozitif sınıf F1 farkı **−0,0002342493**. 1.000 tekrarlı eşlenik şablon-grubu bootstrap %95 aralığı **[−0,0005386458; 0,0000584433]**, yani sıfırı içeriyor. Bu ayrımda ek göstergelerin F1 üstünlüğü gösterilemedi. Bu sonuç kesin kötüleşme, eşdeğerlik veya bütün duygu modellerinin işe yaramadığı iddiası değildir.

Kaydedilen test tahminlerinden ölçümler ayrıca yeniden hesaplandı ve doğrulandı. **25 yazılım testi de geçti; bu test sayısı sınıflandırma başarısı değildir.**

## Veri ve kaynak izi

- Ham veri: **82.486** kayıt; 39.595 etiket 0, 42.891 etiket 1.
- Bir boş içerik ve 409 normalize metin tekrarı çıkarıldı; **82.076** kayıt kaldı. Çelişkili normalize-metin etiketi bulunmadı.
- Eğitim **49.246**, doğrulama **16.415**, test **16.415**. Nihai yeniden eğitim, ilk iki bölmenin toplamı olan **65.661** kayıtla yapıldı.
- Bölmeler arası tam metin hash ve sezgisel şablon hash çakışması **0**. Bu, tüm anlamsal yakın kopyaların giderildiği anlamına gelmez.
- CSV: 106.603.543 bayt; SHA-256 `73a5c5410baf46ed69517fedfd37910f2e93777dc504b73ea483bc16be3a69d1`.
- Veri indirme kaydı: [Actions run 43631449962](https://github.com/DarryHack/kaggle-oltalama-deney-paketi/actions/runs/43631449962).
- Çalıştırılan `experiment.py` Git blob: `ec615297cc7058c3b34afa540ae7a464d8706662`; aynı kodu içeren commit: `cc4b2fa5b98da4e8701a4b1903fcd63d789d87cd`.

## Deney tasarımı

A: 1–2 gram TF-IDF, en çok 50.000 özellik, `min_df=2`, alt-doğrusal terim frekansı. B: aciliyet, korku/tehdit, ödül ve güven/otorite için önceden dondurulmuş İngilizce sözlüklerden 14 gösterge. Her grupta varlık, 100 sözcük başına sıklık ve logaritmik eşleşme sayısı; ayrıca iki ortak-varlık göstergesi. C: aynı metin ve gösterge temsillerinin birleşimi. Bu sözlükler gerçek insan duygusu etiketleri değildir.

Metinler deterministik olarak normalize edilir; HTML temizlenir, yakalanabilen URL/e-postalar maskelenir, metin ilk 20.000 normalize karakterle sınırlandırılır. E-posta bağlantıları ziyaret edilmez. Aynı normalize metnin tekrarları çıkarılır; çelişkili etiketli metin grupları varsa tamamen çıkarılır. Sayı/noktalama varyantlarını birleştiren sezgisel şablon grupları bölmeler arasında paylaşılmaz.

İlk grup-tabakalı 5-kat ayrımından test, kalan verinin ilk 4-kat ayrımından doğrulama seçilir; yaklaşık %60/%20/%20. **Bu bir tek-sabit-ayrım deneyidir, ortalaması alınmış çapraz doğrulama değildir.** Tohum 42'dir. TF-IDF ve StandardScaler yalnızca ilgili eğitim bölümünde öğrenilir.

Tüm modeller `liblinear`, `class_weight=balanced`, `max_iter=3000`, `tol=1e-4` lojistik regresyondur. C={0,1; 1; 10} içinden yalnızca doğrulama macro-F1 ile seçim yapılır; eşitlikte küçük C kullanılır. Seçilenler A=10, B=1, C=10. Seçim sonrası eğitim+doğrulama birleştirilerek yeniden eğitim, ardından ayrılmış testte tek son değerlendirme yapılır. Test eşiği 0,5 olarak sabittir.

AP, average precision'dır; trapez PR-AUC diye sunulmaz. F1 farkı için bootstrap grupları eşlenik yeniden örnekler; aralık sabit modeller ve tek ayrım koşulundadır.

## Yeniden çalıştırma

Python 3.11 veya üstü; ölçülen ortam Python 3.13.5, numpy 2.3.5, pandas 2.2.3, scipy 1.17.0, scikit-learn 1.8.0, matplotlib 3.10.8, threadpoolctl 3.6.0.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python experiment.py --csv /tam/yol/phishing_email.csv --output results/yeni_calisma
```

Alternatif indirme: `python experiment.py --download --output results/yeni_calisma`. Yalnızca belirtilen Kaggle tanımlayıcısı kullanılır. Ağ veya kimlik doğrulama sorunu halinde başka veriyle devam edilmez. Anahtarları veya parolaları koda/GitHub'a yazmayın. Yeni çalışma için boş çıktı klasörü kullanın.

`Kaggle_Deney.ipynb` Kaggle'da aynı veri kümesi Add Input ile eklenip Run All ile çalıştırılabilir. **Depodaki notebook çıktı hücreleri doldurulmamıştır; burada raporlanan gerçek çalışma `experiment.py` modülünün doğrudan çalıştırılmasıdır.** Notebook çalıştırılmış gibi gösterilmemiştir.

## Kanıt dosyaları ve sunum

Bu depoda gerçek sonuç CSV'si ve toplu doğrulama kaydı bulunur. Yazarlara teslim edilen **Oltalama_Gercek_Deney_Kanitlari.zip** ayrıca örnek düzeyi test olasılıkları, ayrım hash kayıtları, çalıştırma manifesti, model seçimi kayıtları, grafikler ve yeniden doğrulama kodunu içerir. Ham e-posta metinleri bu kanıt paketine eklenmez. Güvenilmeyen pickle/joblib dosyalarını açmayın.

Tam metin ve sunum kod satırı içermez; yöntem akışı ve gerçek ölçümleri içerir. Kaynak kod referansı bu deponun gerçek bağlantısıdır. Depo kontrol tarihinde **özel erişimli** olduğundan öğretim elemanı ve dinleyicilerin inceleyebilmesi için gerekli erişim ayrıca sağlanmalıdır. Görünürlük kendiliğinden değiştirilmemiştir.

## Sınırlılıklar

Kaynak pozitif sınıfı spam ve oltalamayı birlikte kapsar; bu oranlar saf kimlik-avı veya gerçek üretim trafiği başarısı değildir. İngilizce sabit sözlükler doğrulanmış duygu etiketleri değildir; olumsuzlama ve bağlamı bütünüyle çözmez. Kaynak metinlerin ön işlenmesi çok sözcüklü örüntü eşleşmelerini etkileyebilir. Tek iç ayrım kaynak-dışı, zamansal veya güncel Türkçe e-posta başarısını kanıtlamaz. Şablon gruplaması tüm yakın kopyaları veya kaynak yanlılığını gidermez. 20.000 karakter sınırı uzun içerikleri kesebilir.

## Kaynaklar

1. Alam, N. A. Phishing Email Dataset. Kaggle. https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
2. Al-Subaiey ve diğerleri (2024). Novel Interpretable and Robust Web-based AI Platform for Phishing Email Detection. https://arxiv.org/abs/2405.11619
3. scikit-learn: StratifiedGroupKFold, TfidfVectorizer, LogisticRegression ve average_precision_score resmi dokümantasyonu. https://scikit-learn.org/stable/

Ham verinin yeniden dağıtımı ve proje lisansı ayrı konulardır; bu güncelleme yeni bir açık kaynak lisansı veya veri yeniden dağıtım izni tanımlamaz.
