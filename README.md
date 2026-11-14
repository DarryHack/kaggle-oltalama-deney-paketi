# Sosyal Mühendislik Metinlerinde Duygu ve Aciliyet Analizi ile Oltalama Tespiti

**Görkem Poyrazoğlu** — ORCID: https://orcid.org/0009-0008-8998-7601  
**Begüm Düger** — ORCID: https://orcid.org/0009-0005-4847-172X  
Gazi University, Information Technologies – Security Program, Ankara, Türkiye

## Mevcut durum — araştırma sonucu değildir

Bu paket yalnızca aşağıdaki veri seti için hazırlanmıştır:

https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset?resource=download

Veri kümesi tanımlayıcısı: `naserabdullahalam/phishing-email-dataset`  
Hedef dosya: `phishing_email.csv`  
Beklenen sütunlar: `text_combined`, `label`

**Hazırlama ortamında ham CSV indirilemedi. Kaggle verisi üzerinde eğitim veya performans değerlendirmesi yapılmadı. Bu pakette doğruluk/F1 sonucu, eğitilmiş araştırma modeli veya tamamlanmış deney iddiası yoktur.** GitHub'a da henüz yüklenmemiştir.

`results/software_tests.txt` ve `results/software_tests.xml` dosyaları 25 yazılım testinin çıktısıdır. Testlerdeki küçük, yapay birim-test girdileri yalnızca programın işlemlerini denetler; Kaggle kayıtları değildir, deney verisine eklenmez ve makalede araştırma sonucu olarak kullanılamaz. `execution_status.json` veri erişiminin durumunu ayrıca kaydeder.

## En kısa çalıştırma: Kaggle Notebook

1. `Kaggle_Deney.ipynb` dosyasını Kaggle Notebook'a aktarın.
2. **Add Input / Add Data** bölümünden tam olarak `naserabdullahalam/phishing-email-dataset` veri setini ekleyin. Benzer isimli başka bir veri seti seçmeyin.
3. CPU ortamında **Run All** çalıştırın. Notebook, bağlı `phishing_email.csv` dosyasını kullanır; bağlı dosya varsa internet erişimi veya GPU gerekmez. Gerekli Python kütüphaneleri ortamda kurulu olmalıdır.
4. Son hücre, yalnızca başarılı deneyden sonra sonuçları ZIP yapar. Notebook'un teslim edilen sürümündeki çıktı hücreleri bilerek boştur; çalıştırılmış görüntüsü verilmemiştir.

Veri bağlı değilse notebook durur. Sentetik veri, başka veri seti veya başkasının sonuçları otomatik yedek olarak kullanılmaz.

## Terminalden çalıştırma

Python 3.11 veya üstü önerilir. Paket kökünde:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python experiment.py --download --output results/kaggle_run
```

`--download`, yalnızca yukarıdaki Kaggle tanımlayıcısını `kagglehub` ile indirir. Kaggle kimlik doğrulaması veya internet kısıtı varsa program açık hata verir. Şifre veya API anahtarlarını kaynak koda, GitHub'a ya da sohbet mesajlarına yapıştırmayın.

Dosya önceden indirilmişse:

```bash
python experiment.py --csv /tam/yol/phishing_email.csv --output results/kaggle_run
```

Yeni çalıştırmada yeni bir çıktı klasörü kullanılır; eski ölçümler sessizce ezilmez. İsteğe bağlı `--save-models`, modelleri yerel `models.joblib` dosyasına kaydeder. Güvenilmeyen pickle/joblib dosyalarını açmayın. Kaynak veri CSV'leri ve model dosyaları GitHub'a otomatik yüklenmez.

## Deney tasarımı

**A — Metin:** 1–2 gram TF-IDF ve lojistik regresyon.  
**B — Göstergeler:** Aciliyet, korku/tehdit, ödül ve güven/otorite sözcük gruplarına ait 14 açıklanabilir özellik ve lojistik regresyon.  
**C — Birleşik:** Aynı TF-IDF temsili ile aynı 14 gösterge ve lojistik regresyon.

Gösterge sözlükleri `experiment.py` içinde dondurulmuştur. Bunlar araştırmacı tanımlı İngilizce sözcük listeleridir; doğrulanmış duygu etiketleri, insan duygu ölçümü, duygusal zekâ modeli veya nedensel psikoloji ölçümü değildir. Olumsuzlama, ironi ve bağlamın tamamını çözmezler. Ana hipotez C'nin A'ya ek katkısıdır; iyileşme çıkması zorunlu değildir.

### Veri bütünlüğü ve ayrım

Yalnızca birleşik `phishing_email.csv` kullanılır; aynı paketteki alt kaynaklar bu dosyaya yeniden eklenerek çift sayım yapılmaz. Eksik/boş kayıtlar ve aynı normalize metnin tekrarları sayılır. Aynı normalize metne farklı etiketler verilmişse bu çelişkili grubun bütün kayıtları çıkarılır. 0/1 dışı etiketler görülürse program durur.

URL/e-posta metinleri maskelenir; HTML içerik metin olarak temizlenir. Hiçbir e-posta bağlantısına istek gönderilmez. Normalize metinler ilk 20.000 karakterle sınırlandırılır. Bu tercih veri denetiminde kaydedilir ve uzun iletilerin sonundaki ipuçlarını kaybettirebilir.

URL, e-posta, sayı ve noktalama değişikliklerine duyarsız bir şablon özeti üretilir. Aynı şablon grubunun eğitim, doğrulama ve test bölmeleri arasında dağılması engellenir. Bu, anlamı benzer tüm metinlerin ayıklanması anlamına gelmez.

İlk grup-tabakalı 5-kat ayrımının bir katı test için ayrılır. Kalan geliştirme verisinin ilk grup-tabakalı 4-kat ayrımından bir katı doğrulama olur: yaklaşık %60 eğitim, %20 doğrulama, %20 test. Grup boyutlarından dolayı oranlar tam eşit olmayabilir. Bu işlemler **5-kat çapraz doğrulama sonucu raporlama** değildir; yalnızca tek sabit ayrımı üretir. Rastgele tohum 42'dir.

### Model seçimi ve değerlendirme

C değerleri 0,1; 1 ve 10'dur. Her model için doğrulama macro-F1'i en yüksek olan C seçilir; eşitlikte küçük C kullanılır. Tahmin eşiği 0,5 ve sınıf ağırlığı `balanced` olarak sabittir. TF-IDF sözlüğü ve sayısal özellik ölçeklemesi önce yalnızca eğitim bölümünde öğrenilir. Seçimden sonra eğitim+doğrulama üzerinde yeniden öğrenilir; bağımsız test bölümünde tek son değerlendirme yapılır.

Ölçüler: accuracy, balanced accuracy, pozitif sınıf precision/recall/F1, macro-F1, ROC-AUC, **average precision (AP)**, yanlış pozitif oranı ve TN/FP/FN/TP. AP, trapez yöntemiyle hesaplanmış PR-AUC diye adlandırılmaz.

C−A pozitif sınıf F1 farkı için aynı test şablon gruplarını birlikte yeniden örnekleyen 1.000 tekrarlı eşlenik bootstrap ve %95 yüzdelik aralık hesaplanır. Aralık tek ayrım ve sabit modeller koşulundadır; dış veri başarısı veya nedensellik kanıtı değildir.

## Gerçek çalıştırma sonrasında üretilecek dosyalar

- `run_manifest.json`: tamamlanma durumu, veri SHA-256 özeti, boyut, ayarlar, kütüphane sürümleri ve ayrım sayıları.
- `data_audit.json`, `split_manifest.csv`: veri temizlik sayıları ve hash temelli ayrım denetimi.
- `validation_scores.csv`, `metrics.csv`, `metrics.json`, `classification_reports.json`: model seçimi ve gerçek test ölçümleri.
- `test_predictions.csv`, `paired_f1_comparison.json`: e-posta metni içermeyen örnek düzeyi test tahminleri ve eşlenik karşılaştırma.
- `cue_descriptives_test.csv`, `cue_model_coefficients.csv`: göstergelerin açıklayıcı çıktıları; katsayılar nedensellik ifade etmez.
- `f1_comparison.png`, `confusion_*.png`: gerçek ölçümlerden oluşturulan grafikler.

**Kullanım şartı:** `run_manifest.json` içindeki `status` değeri `completed` olmadıkça klasördeki kısmi çıktılar tamamlanmış deney diye raporlanmamalıdır.

## Tam metin ve sunum kuralları

Hocanın talimatına göre tam metne ve slaytlara kaynak kod eklenmez. Kod bu depoda tutulur. Yöntem metni için `docs/METOT_VE_RAPORLAMA.md`, akış için `docs/akis.mmd` hazırlanmıştır. Akış görseli kullanılabilir; sonuç slaytları ancak gerçek çalıştırma tamamlandığında hazırlanmalıdır.

Depo henüz oluşturulmadığından gerçek bir GitHub bağlantısı verilmemiştir. Bu klasör bir depoya yüklendikten sonra deponun **gerçek URL'si ve mümkünse commit kimliği** Materyal ve Metot bölümüne ve sunumun son slaydına eklenmelidir. Yapılmamış yükleme veya temsili URL, mevcut depo diye gösterilmemelidir.

## Bilimsel sınırlılıklar

Verinin etiket sözlüğü ham kaynakla teyit edilmelidir. Pozitif sınıf spam, dolandırıcılık ve oltalamayı birlikte içeriyorsa başarı saf oltalama başarısı olarak adlandırılmaz. İngilizce sözlükler Türkçe saldırılarda doğrulanmış değildir; dil dağılımı bu hazırlama aşamasında ölçülmemiştir. Şablon gruplaması tüm yakın kopyaları ve kaynak yanlılığını gidermez. Zamansal, kaynak-dışı veya modern saldırı değerlendirmesi bu protokolde bulunmaz. Veri üzerinde yüksek iç test skoru elde edilse bile gerçek e-posta akışında aynı başarı varsayılamaz.

## Kaynaklar

1. Naser Abdullah Alam. *Phishing Email Dataset*. Kaggle. https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
2. scikit-learn, *StratifiedGroupKFold*. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
3. scikit-learn, *TfidfVectorizer*. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
4. scikit-learn, *LogisticRegression*. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
5. scikit-learn, *average_precision_score*. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html

Veri kümesinin yeniden dağıtım koşulları bağımsız olarak doğrulanmadığı için ham e-postalar bu pakete eklenmemiştir. Kütüphane lisansları kendi projelerine aittir; proje için açık kaynak lisansı seçimi yazarlara bırakılmıştır.
