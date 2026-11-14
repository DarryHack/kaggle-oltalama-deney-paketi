# Sonuçlar — gerçek deney tamamlandı

**14 Kasım 2026:** `naserabdullahalam/phishing-email-dataset`, sürüm 1, `phishing_email.csv` üzerinde eğitim ve ayrılmış test değerlendirmesi gerçekleştirildi.

- [Gerçek test ölçümleri](real_run_20261114/metrics.csv): üç model, aynı 16.415 test e-postası.
- [Doğrulanmış deney kaydı](real_run_20261114/verified_results.json): dosya SHA-256, kaynak sürümü, temizleme sayıları, ayrım, ayarlar, eşlenik bootstrap ve sınırlılıklar.
- [Çalıştırma durumu](execution_status.json): tamamlanmış gerçek deney durumunu belirtir.
- `software_tests.txt` ve `software_tests.xml`: 25 yazılım testi; araştırma başarısı değildir. Küçük yapay birim-test girdileri araştırma verisine katılmamıştır.

A metin modelinin pozitif F1'i 0,991429; B gösterge modelinin 0,469992; C birleşik modelin 0,991195'tir. C−A farkı −0,000234; %95 eşlenik grup bootstrap aralığı sıfırı içerir. Ek göstergelerin F1 üstünlüğü bu tek iç ayrımda gösterilememiştir.

Tam örnek düzeyi test tahminleri, ayrım hash'leri, manifestler ve yeniden hesaplama kodu yazarlara teslim edilen `Oltalama_Gercek_Deney_Kanitlari.zip` içindedir. Ham e-posta gövdeleri depoya veya bu kanıt paketine eklenmemiştir.

Pozitif sınıf spam ve oltalamayı birleştirir; yüksek iç test sonucu saf oltalama, Türkçe e-posta veya gerçek üretim trafiği başarısı olarak sunulmamalıdır.
