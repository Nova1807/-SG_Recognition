# ÖGS Gebärden-Erkennung

Ein Programm zur Erkennung von Österreichischer Gebärdensprache (ÖGS) über Webcam.

## Features

- **Automatischer Video-Download**: Sucht und lädt Gebärdenvideos von [gebaerden-archiv.at](https://gebaerden-archiv.at/search) herunter
- **Live-Webcam-Erkennung**: Erkennt Gebärden über den Live-Webcam-Feed mittels MediaPipe und einem trainierten Modell
- **Aufnahme-Modus**: Wenn nicht genug Trainingsvideos vorhanden sind, kann man die Gebärde sehen und selbst aufnehmen
- **Konfigurierbar**: Akzeptierte Gebärden werden aus einer `gebaerden.txt` Datei gelesen

## Installation

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

## Verwendung

### 1. Gebärden definieren

Bearbeite `gebaerden.txt` und füge die gewünschten Gebärden hinzu (eine pro Zeile):

```
Hallo
Danke
Bitte
Ja
Nein
```

### 2. Videos herunterladen

```bash
python -m src.main download
```

Lädt automatisch Videos von gebaerden-archiv.at für alle Gebärden in `gebaerden.txt`.

### 3. Trainings-Landmarks extrahieren

```bash
python -m src.main extract
```

Extrahiert Hand- und Körper-Landmarks aus allen heruntergeladenen Videos und Aufnahmen.

### 4. Modell trainieren

```bash
python -m src.main train
```

Trainiert ein Erkennungsmodell basierend auf den extrahierten Landmarks.

### 5. Live-Erkennung starten

```bash
python -m src.main recognize
```

Öffnet die Webcam und erkennt Gebärden in Echtzeit.

### 6. Aufnahme-Modus

```bash
python -m src.main record
```

Zeigt Gebärden mit zu wenig Trainingsdaten an und ermöglicht eigene Aufnahmen.

## Projektstruktur

```
oegs-sign-recognition/
├── src/
│   ├── __init__.py
│   ├── main.py              # Haupteinstiegspunkt
│   ├── config.py             # Konfiguration
│   ├── scraper.py            # Video-Scraper für gebaerden-archiv.at
│   ├── landmark_extractor.py # MediaPipe Landmark-Extraktion
│   ├── trainer.py            # Modell-Training
│   ├── recognizer.py         # Live-Erkennung
│   └── recorder.py           # Aufnahme-Modus
├── data/
│   ├── videos/               # Heruntergeladene Videos (pro Gebärde)
│   ├── recordings/           # Eigene Aufnahmen (pro Gebärde)
│   └── landmarks/            # Extrahierte Landmarks
├── models/                   # Trainierte Modelle
├── gebaerden.txt             # Liste der akzeptierten Gebärden
├── requirements.txt
└── README.md
```

## Abhängigkeiten

- Python 3.10+
- OpenCV
- MediaPipe
- scikit-learn
- BeautifulSoup4
- requests

## Lizenz

MIT
