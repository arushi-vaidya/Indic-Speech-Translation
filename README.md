# 🎙️ Indic Speech Translation

**Indic Speech Translation** is a tool designed to convert English speech to **Kannada speech**. It performs **noise reduction**, **language translation**, and **text-to-speech synthesis** using state-of-the-art models provided by [AI4Bharat](https://ai4bharat.org/).

This project can be useful in voice-based interfaces, accessibility tools, and multilingual communication systems.

---

## 🚀 Features

* ✅ Noise suppression using a spectral gating algorithm
* ✅ English to Kannada translation using **AI4Bharat’s IndicTrans** models
* ✅ Kannada Text-to-Speech (TTS) synthesis
* ✅ Lightweight and modular pipeline

---

## 🛠️ Prerequisites

* Python 3.8 or higher
* Git
* Internet connection (to download model checkpoints)

---

## 📁 Project Structure

```
Indic-Speech-Translation/
├── main2.py                 # Main pipeline script
├── requirements.txt         # Python dependencies
├── README.md                # Documentation
├── Noise_Supression/        # Custom noise reduction module
├── models/                  # Pretrained models (or downloaded on first run)
└── ...                      # Other support files
```

---

## 🧑‍💻 Setup Instructions

1. **Clone the AI4Bharat IndicTrans Toolkit** (if not already available):

   ```bash
   git clone https://github.com/AI4Bharat/IndicTrans.git
   ```

2. **Copy `main2.py` into the IndicTrans directory**:

   ```bash
   cp main2.py IndicTrans/
   cd IndicTrans
   ```

3. **Create a virtual environment and activate it**:

   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

4. **Install all required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the speech translation pipeline**:

   ```bash
   python3 main2.py
   ```

---

## 🧪 What Happens Under the Hood

1. 🎧 **Input Audio** is passed through a **spectral gating** algorithm to reduce background noise.
2. 📝 The cleaned **English speech** is transcribed (if needed) and translated to **Kannada text** using **IndicTrans**.
3. 🔊 The translated **Kannada text** is then converted to natural-sounding **speech** using a **Kannada TTS model**.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

