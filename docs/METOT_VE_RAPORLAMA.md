# Materyal ve Metot için uygulama notu

**Durum:** Bu metin çalıştırılabilir protokolü açıklar; tamamlanmış deney anlatımı değildir. Kod veya öğretim örneği sonuçları araştırma bulgusu olarak eklenmemiştir.

## Veri kaynağı

Seçilen kaynak Naser Abdullah Alam'ın Kaggle üzerindeki Phishing Email Dataset veri kümesidir. Deney programı yalnızca bu paketteki `phishing_email.csv` dosyasını kullanacak şekilde hazırlanmıştır. Veri boyutu, sınıf dağılımı, yinelenen kayıt sayıları ve dosyanın SHA-256 özeti ham dosya elde edilince doğrudan hesaplanacaktır; burada başka çalışmaların sayıları kendi ölçümümüz gibi verilmemektedir.

## İşlem basamakları

Metin temizliği ve duyarsızlaştırma; çelişkili/tekrarlı kayıtların denetimi; aynı şablonların gruplandırılması; yaklaşık %60/%20/%20 eğitim/doğrulama/test ayrımı; TF-IDF ve sözlük tabanlı göstergelerin çıkarılması; üç lojistik regresyon modelinin karşılaştırılması; bağımsız test değerlendirmesi; eşlenik fark analizi ve sonuçların raporlanması.

E-posta metinleri ve içeriklerindeki URL'ler yürütülmez veya ziyaret edilmez. Özellik çıkarımının öğrenilen parçaları test kümesine bakmadan kurulacaktır. Lojistik regresyonun C parametresi doğrulama bölümünde seçilecektir.

## Duygu/aciliyetin işlemsel tanımı

Aciliyet, korku/tehdit, ödül ve güven/otorite sözcükleri için varlık, 100 sözcük başına sıklık ve logaritmik eşleşme sayısı hesaplanacaktır. Aciliyet-korku ve aciliyet-ödül ortak varlığı iki ilave özellik oluşturur. Toplam 14 özellik vardır. Bunlar doğrulanmış psikolojik ölçümler değildir; araştırma hipotezine yönelik açıklanabilir vekil göstergelerdir.

## Karşılaştırma

A modeli TF-IDF; B modeli yalnızca göstergeler; C modeli birleşik özellikler kullanır. Sonuçların A ve C arasındaki farkı, hipotezin sınanmasında temel karşılaştırmadır. Başarı artışı olmazsa veya farkın güven aralığı sıfırı kapsarsa bu durum açıkça belirtilecektir.

## Kaynak kodun erişilebilirliği

Kaynak kod ayrı depo paketinde hazırlanmıştır. GitHub'a henüz yüklenmediği için bu aşamada gerçek depo URL'si yoktur. Yüklemeden sonra bu paragraf, doğrulanmış URL ve commit kimliğini içerecek şekilde güncellenmelidir. Depo açılmadan temsili bir bağlantı kaynakça gibi gösterilmemelidir. Kodun tamamı tam metne veya sunuma eklenmez.

## Bulgular için tamamlanma kontrolü

`run_manifest.json` dosyasının `completed` olması; veri SHA-256 değerinin kaydedilmesi; test bölümünün eğitim ve doğrulamadan ayrılmış olması; üç modelin test tahminlerinin saklanması ve `metrics.csv` ile tutarlı olması gereklidir. Bu koşullar karşılanmadan taslaktaki "deney tamamlanmadı" ifadeleri kaldırılmamalıdır.

AP ile PR-AUC aynı hesaplama adı altında kullanılmaz. Kod AP raporlar. Etiket 1 yalnızca oltalama anlamına gelmiyorsa görevin adı ve sonuçların kapsamı buna uygun sınırlandırılır.
