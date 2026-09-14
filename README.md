# TinyML-Anomaly-Detection-STM32-NUCLEO-H753ZI

Real-time temperature anomaly detection using a quantized autoencoder deployed bare-metal on an STM32H753ZI (Nucleo-H753ZI). The full pipeline goes from raw sensor data → Python training (Colab) → TFLite int8 → STM32 Edge AI Platform → C inference with LED alert.

---

## How It Works

1. The internal temperature sensor (ADC3) samples at 10 Hz
2. A sliding window of 10 samples is normalized and fed to the autoencoder
3. The model reconstructs the input; high MSE = anomaly
4. LD1 (green LED) lights up on anomaly; results stream over UART at 115200 baud

---

## Repository Structure

```
├── data/
│   └── temp_readings.txt          # Raw sensor readings (~1000 samples, ~53°C idle)
├── training/
│   └── train_colab.ipynb          # Autoencoder training + TFLite int8 export
├── model/
│   └── temp_anomaly_model.tflite  # Quantized int8 TFLite model (~X KB)
├── firmware/
│   ├── main.c
│   └── app_x-cube-ai.c   # Inference loop, ADC read, MSE + decision
└── README.md
```

---

## Model

| Parameter | Value |
|---|---|
| Architecture | Autoencoder (10→8→4→8→10) |
| Training data | ~1000 ADC samples, internal temp sensor |
| Normalization | mean=53.6158, std=0.2756 |
| Anomaly threshold | MSE > 1.309446 (95th percentile of training errors) |
| Quantization | float32 → int8 (TFLite DEFAULT optimization) |

---

## Training

Open `training/train_colab.ipynb` in Google Colab. The notebook:
- Parses raw readings (accessible from `data/temp_readings.txt`)
- Trains a dense autoencoder (50 epochs, Adam, MSE loss)
- Exports to TFLite int8 with representative dataset calibration
- Validates the quantized model with normal and injected anomaly windows

---

## Firmware

**Toolchain:** STM32CubeIDE, HAL, STM32 Edge AI Platform (X-CUBE-AI)  
**Target:** NUCLEO-H753ZI (STM32H753ZI @ 64 MHz via HSI)

Key files:
- `app_x-cube-ai.c` — inference loop: ADC read → normalize → `stai_network_run()` → MSE → GPIO/UART output
- `main.c` — peripheral init (ADC3, USART3, GPIO), calls `MX_X_CUBE_AI_Process()` in the main loop

UART output (115200 8N1 via ST-Link USB):
```
Normal.  temp=53.61 error=0.0023
ANOMALY! temp=55.10 error=1.4821
```

---

## Known Limitations & Future Work

The training dataset was collected under a single idle condition (~53°C), resulting in very low variance (std ≈ 0.28°C). The model performs well within that operating point but will flag legitimate temperature changes as anomalies.

Planned improvements:
- Collect data across varied thermal conditions (idle, under load, different ambient temps)
- Use an external precision sensor (DS18B20, NTC thermistor) for higher-quality data
- Retrain with broader normal distribution to reduce false positive rate

---

## Requirements

**Python (training):**
```
tensorflow>=2.13
numpy
```

**Hardware:**
- NUCLEO-H753ZI
- STM32CubeIDE + X-CUBE-AI pack
