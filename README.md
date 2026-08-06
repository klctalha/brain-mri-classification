# Brain MRI Classification / Beyin MR Sınıflandırması

Bu depo, beyin MR görüntülerinde tümör varlığını ikili olarak sınıflandırmak için MATLAB ile geliştirilmiş bir derin öğrenme çalışmasıdır. Proje, sıfırdan tanımlanan bir CNN ile transfer öğrenme tabanlı MobileNetV2 modelini; Adam, RMSprop ve SGDM eniyileyicileri altında eğitir, test eder ve karşılaştırır.

This repository contains a MATLAB deep-learning study for binary tumor classification in brain MRI images. It trains, evaluates, and compares a custom CNN and a transfer-learning-based MobileNetV2 model using the Adam, RMSprop, and SGDM optimizers.

> [!IMPORTANT]
> Bu çalışma araştırma ve eğitim amaçlıdır; klinik tanı aracı değildir. / This project is intended for research and education only; it is not a clinical diagnostic tool.

## Türkçe

### Proje özeti

Ana betik [`proje_kodlar.m`](proje_kodlar.m) aşağıdaki iş akışını tek dosyada yürütür:

1. `archive/brain_tumor_dataset` altındaki görüntüleri klasör adlarından etiketler.
2. Veriyi sınıf bazında rastgele %70 eğitim, %15 doğrulama ve %15 test olarak ayırır.
3. Görüntüleri 224 × 224 piksele getirir; gri tonlamalı görüntüleri üç kanala dönüştürür.
4. Eğitim verisine döndürme, öteleme ve ölçekleme artırımları uygular.
5. Özel CNN ve önceden eğitilmiş MobileNetV2 modellerini üç farklı eniyileyiciyle eğitir.
6. Accuracy, precision, sensitivity/recall ve F1-score metriklerini hesaplar; karışıklık matrisleri ve karşılaştırma grafikleri üretir.
7. Modelleri, eğitim bilgilerini, hiperparametre tablolarını ve MATLAB figürlerini kaydeder.

### Modeller

| Model | Yapı | Eniyileyiciler |
|---|---|---|
| Özel CNN | 3 evrişim bloğu (32/64/128 filtre), batch normalization, ReLU, max pooling, 128 birimli tam bağlı katman ve %50 dropout | Adam, RMSprop, SGDM |
| MobileNetV2 | ImageNet üzerinde önceden eğitilmiş ağ; son sınıflandırma katmanları iki sınıfa uyarlanır | Adam, RMSprop, SGDM |

Her deneyde başlangıç öğrenme oranı `1e-4`, azami epoch sayısı `20`, mini-batch boyutu `16`, doğrulama sıklığı `30` ve erken durdurma sabrı `5` olarak tanımlanmıştır. SGDM için momentum `0.9`'dur. En düşük doğrulama kaybına sahip ağ saklanır.

### Veri kümesi düzeni

Veri kümesi depoya dahil değildir. Betik aşağıdaki yapıyı bekler:

```text
brain-mri-classification/
├── proje_kodlar.m
└── archive/
    └── brain_tumor_dataset/
        ├── no/
        │   └── ... görüntüler
        └── yes/
            └── ... görüntüler
```

- `no`: tümör bulunmayan görüntüler
- `yes`: tümör bulunan görüntüler

Çalışma sırasında yeniden boyutlandırılmış kopyalar `archive/brain_tumor_dataset_resized/{train,validation,test}/{no,yes}` altında oluşturulur. Kaynak veri setinin kullanım koşullarını ve lisansını ayrıca kontrol edin.

### Gereksinimler

- MATLAB (güncel bir sürüm önerilir)
- Deep Learning Toolbox
- Image Processing Toolbox
- Deep Learning Toolbox Model for MobileNet-v2 Network destek paketi
- İsteğe bağlı fakat önerilen: uyumlu NVIDIA GPU ve Parallel Computing Toolbox

MobileNetV2 destek paketini MATLAB Add-On Explorer üzerinden kurabilirsiniz. GPU yoksa `ExecutionEnvironment = 'auto'` nedeniyle eğitim CPU üzerinde devam eder ancak belirgin biçimde daha uzun sürebilir.

### Kurulum ve çalıştırma

1. Depoyu klonlayın ve MATLAB çalışma klasörünü proje köküne ayarlayın.
2. Veri kümesini yukarıdaki dizin yapısına göre yerleştirin.
3. MobileNetV2 destek paketinin kurulu olduğunu doğrulayın:

   ```matlab
   net = mobilenetv2;
   ```

4. Betiği MATLAB Editor'da açıp bölüm bölüm çalıştırın veya tamamını yürütün:

   ```matlab
   run('proje_kodlar.m')
   ```

Betik uzun süren altı eğitim deneyi yürütür. Daha kontrollü kullanım için `%%` ile başlayan bölümlerin sırasını koruyarak bölüm bölüm çalıştırılması önerilir. İlk veri hazırlama adımını yeniden çalıştırmadan önce üretilen `brain_tumor_dataset_resized` klasörünü kontrol edin; mevcut dosyalar aynı adlarla yeniden yazılabilir.

### Veri artırma

Eğitim setine çevrimiçi olarak şu rastgele dönüşümler uygulanır:

- Döndürme: −10° ile +10°
- X/Y öteleme: −5 ile +5 piksel
- X/Y ölçekleme: 0.9 ile 1.1

Doğrulama ve test verilerine artırma uygulanmaz.

### Çıktılar

Çalıştırma sırasında aşağıdaki dosyalar oluşturulur (Git tarafından yok sayılır):

- `*_best_model.mat`: en iyi doğrulama kaybına sahip ağ
- `*_hyperparameters.csv`: deney hiperparametreleri
- `*_trainNetworkProject.mat`: ağ, eğitim geçmişi ve eğitim seçenekleri
- `mobilenetv2_sgdm_test_results.mat`: MobileNetV2-SGDM test metrikleri
- `trainNetworkProject/*.fig`: veri dağılımı ve model karşılaştırma grafikleri

### Betikte raporlanan sonuçlar

Aşağıdaki test doğrulukları karşılaştırma grafikleri için kaynak kodda sabit değer olarak yer almaktadır; bu depoda eğitim çıktıları veya veri kümesi bulunmadığından bağımsız olarak doğrulanmamıştır.

| Model | Adam | RMSprop | SGDM |
|---|---:|---:|---:|
| Özel CNN | %90.41 | %88.15 | %86.97 |
| MobileNetV2 | %95.95 | %93.24 | %94.59 |

Veri ayrımı rastgele yapıldığı ve sabit bir rastgelelik tohumu (`rng`) tanımlanmadığı için her çalıştırmada sonuçlar değişebilir. Güncel deney sonuçları için betiğin test aşamasında hesapladığı metrikleri esas alın.

### Bilinen noktalar

- Betik monolitiktir ve bazı değişken adlarını farklı deneylerde yeniden kullanır; bölümleri sıra dışı çalıştırmak eksik veya eski çalışma alanı değişkenlerine yol açabilir.
- `trainNetworkProject` klasörü betiğin ilerleyen kısmında oluşturulur. Temiz bir çalışma alanında daha erken gerçekleşen `savefig` çağrıları için klasörü önceden oluşturmak gerekebilir.
- Karışıklık matrisi hesabı ikinci kategoriyi pozitif sınıf kabul eder. Klasör/etiket sırası değişirse `TP`, `FP`, `FN` ve `TN` eşlemesini doğrulayın.
- Precision veya recall paydası sıfır olduğunda metrikler `NaN` olabilir; betikte sıfıra bölme koruması bulunmaz.
- Kaynak dosyada Türkçe karakterlerin kodlaması bazı editörlerde bozuk görünebilir. Dosyayı uygun kodlamayla açın.
- Projede lisans dosyası yoktur; yeniden kullanım koşulları belirtilmemiştir.

---

## English

### Project overview

The main script, [`proje_kodlar.m`](proje_kodlar.m), performs the complete workflow in one file:

1. Loads images from `archive/brain_tumor_dataset` and derives labels from folder names.
2. Performs a stratified random split: 70% training, 15% validation, and 15% testing.
3. Resizes images to 224 × 224 and converts grayscale images to three channels.
4. Applies random augmentation to the training data.
5. Trains a custom CNN and a pretrained MobileNetV2 with three optimizers.
6. Computes accuracy, precision, sensitivity/recall, and F1-score; generates confusion matrices and comparison charts.
7. Saves trained models, training information, hyperparameter tables, and MATLAB figures.

### Models

| Model | Architecture | Optimizers |
|---|---|---|
| Custom CNN | 3 convolution blocks (32/64/128 filters), batch normalization, ReLU, max pooling, a 128-unit fully connected layer, and 50% dropout | Adam, RMSprop, SGDM |
| MobileNetV2 | ImageNet-pretrained network with its final classification layers replaced for two classes | Adam, RMSprop, SGDM |

All experiments use an initial learning rate of `1e-4`, up to `20` epochs, a mini-batch size of `16`, a validation frequency of `30`, and an early-stopping patience of `5`. SGDM uses `0.9` momentum. The network with the lowest validation loss is retained.

### Dataset layout

The dataset is not included in this repository. The script expects:

```text
brain-mri-classification/
├── proje_kodlar.m
└── archive/
    └── brain_tumor_dataset/
        ├── no/
        │   └── ... images
        └── yes/
            └── ... images
```

- `no`: MRI images without a tumor
- `yes`: MRI images with a tumor

During execution, resized copies are written to `archive/brain_tumor_dataset_resized/{train,validation,test}/{no,yes}`. Check the original dataset's license and usage terms separately.

### Requirements

- MATLAB (a recent release is recommended)
- Deep Learning Toolbox
- Image Processing Toolbox
- Deep Learning Toolbox Model for MobileNet-v2 Network support package
- Optional but recommended: a compatible NVIDIA GPU and Parallel Computing Toolbox

Install the MobileNetV2 support package through MATLAB Add-On Explorer. Without a GPU, `ExecutionEnvironment = 'auto'` allows training to fall back to the CPU, but it may take substantially longer.

### Setup and usage

1. Clone the repository and set the MATLAB current folder to the project root.
2. Place the dataset in the directory structure shown above.
3. Verify that MobileNetV2 is available:

   ```matlab
   net = mobilenetv2;
   ```

4. Open the script in MATLAB and run it section by section, or execute the full script:

   ```matlab
   run('proje_kodlar.m')
   ```

The script runs six potentially long training experiments. For better control, run the `%%` sections in order. Before rerunning data preparation, inspect the generated `brain_tumor_dataset_resized` directory because existing files with matching names may be overwritten.

### Data augmentation

The training set receives these online random transformations:

- Rotation: −10° to +10°
- X/Y translation: −5 to +5 pixels
- X/Y scaling: 0.9 to 1.1

No augmentation is applied to validation or test data.

### Outputs

Execution produces the following files, all ignored by Git:

- `*_best_model.mat`: network with the best validation loss
- `*_hyperparameters.csv`: experiment hyperparameters
- `*_trainNetworkProject.mat`: network, training history, and training options
- `mobilenetv2_sgdm_test_results.mat`: MobileNetV2-SGDM test metrics
- `trainNetworkProject/*.fig`: dataset-distribution and model-comparison figures

### Results reported in the script

The following test accuracies are hard-coded in the source for comparison plots. They have not been independently verified because neither the dataset nor generated training artifacts are committed to this repository.

| Model | Adam | RMSprop | SGDM |
|---|---:|---:|---:|
| Custom CNN | 90.41% | 88.15% | 86.97% |
| MobileNetV2 | 95.95% | 93.24% | 94.59% |

The data split is randomized and no fixed random seed (`rng`) is set, so results may differ between runs. Use the metrics calculated during the test sections as the authoritative results for a new experiment.

### Known considerations

- The script is monolithic and reuses variable names between experiments. Running sections out of order may leave missing or stale workspace variables.
- The `trainNetworkProject` directory is created late in the script. On a clean run, it may need to be created before earlier `savefig` calls.
- Confusion-matrix calculations assume that the second category is the positive class. Verify the `TP`, `FP`, `FN`, and `TN` mapping if label ordering changes.
- Metrics may become `NaN` when a precision or recall denominator is zero; the script does not currently guard against division by zero.
- Turkish characters in the source may appear garbled in editors using an incompatible text encoding.
- No license file is present, so reuse terms are currently unspecified.

## Repository contents

```text
.
├── .gitignore
├── README.md
└── proje_kodlar.m
```

## License

No license has been provided. Unless a license is added, standard copyright restrictions apply.
