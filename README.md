# İzmir ihracatı ve İngiltere iş çevrimi

Streamlit uygulaması; İzmir ihracatını doğrusal trendden arındırıp oranı 100 tabanlı endekse çevirir, ardından merkezlenmiş 3 yıllık hareketli ortalamayı bir kez uygular. NBER’in çeyreklik İngiltere iş çevrimi endeksini yıllık ortalamaya çevirerek aynı grafikte karşılaştırır.

## Yerel çalıştırma

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Varsayılan veri dosyası `Izmir_UK_NBER_raw_1864_1914.csv` uygulamayla aynı klasörde bulunmalıdır. Uygulama içinden başka bir CSV de yüklenebilir.

## Streamlit Community Cloud

`app.py`, `requirements.txt` ve `Izmir_UK_NBER_raw_1864_1914.csv` dosyalarını bir GitHub deposuna koyun. Streamlit Community Cloud’da bu depoyu seçip ana dosya olarak `app.py` gösterin.

## Veri notu

CSV; İzmir ihracatını sterlin cinsinden yıllık ham değer olarak, NBER serisini ise ham çeyreklik sütunlar halinde saklar. NBER yıllık ortalaması uygulamada yeniden hesaplanır. 1914 yalnızca iki çeyrek içerdiğinden tam yıl ortalaması değildir. NBER serisi İngiltere GSYH’si değil, iş çevrimi endeksidir.
