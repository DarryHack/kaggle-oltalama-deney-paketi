# Bu klasörün mevcut içeriği

Bu pakette **Kaggle üzerinde ölçülmüş araştırma sonucu yoktur**.

- `software_tests.txt` ve `software_tests.xml`: gerçek ortamda çalıştırılmış 25 yazılım birim testinin kayıtlarıdır.
- `execution_status.json`: ham veri indirilemediğini ve gerçek eğitim/test yapılmadığını belirtir.

Birim testleri doğrulamak için kullanılan el yapımı küçük girdiler Kaggle örnekleri değildir. Testlerin geçmesi modelin oltalama başarısını göstermez.

`python experiment.py ... --output results/kaggle_run` gerçek dosya ile başarılı çalıştıktan sonra deney çıktıları `kaggle_run/` altında oluşturulur. Mevcut dosyalara bakarak doğruluk/F1 oranı çıkarılmamalıdır.
