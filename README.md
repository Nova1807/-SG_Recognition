# ÖGS Gebärden-Erkennung

Ein Programm zur Erkennung von Österreichischer Gebärdensprache (ÖGS) über Webcam mit visuellem Web-Interface.

## Features

- **Web-Interface**: Visuelles Dashboard mit allen Funktionen im Browser
- **Automatischer Video-Download**: Sucht und lädt Gebärdenvideos von [gebaerden-archiv.at](https://gebaerden-archiv.at/search) herunter
- **Live-Webcam-Erkennung**: Erkennt Gebärden über den Live-Webcam-Feed mittels MediaPipe und einem trainierten Modell
- **Aufnahme-Modus**: Wenn nicht genug Trainingsvideos vorhanden sind, kann man das Referenzvideo sehen und selbst aufnehmen
- **Konfigurierbar**: Akzeptierte Gebärden werden aus einer `gebaerden.txt` Datei gelesen

## Installation

```bash
git clone https://github.com/Nova1807/-SG_Recognition.git
cd -SG_Recognition
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# oder: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

## Schnellstart

```bash
python3 -m src.main start
```

Öffnet das Web-Interface unter `http://localhost:5000`. Dort kannst du alles über den Browser steuern:

1. **Dashboard**: Übersicht aller Gebärden und Pipeline-Steuerung
2. **Erkennung**: Live-Webcam-Erkennung mit Verlauf
3. **Aufnahme**: Referenzvideo ansehen und eigene Gebärden aufnehmen

## Gebärden definieren

Bearbeite `gebaerden.txt` und füge die gewünschten Gebärden hinzu (eine pro Zeile):

```
Hallo
Danke
Bitte
Ja
Nein
```

## CLI-Befehle (alternativ)

```bash
python3 -m src.main start      # Web-Interface starten (empfohlen)
python3 -m src.main download   # Videos herunterladen
python3 -m src.main extract    # Landmarks extrahieren
python3 -m src.main train      # Modell trainieren
python3 -m src.main recognize  # Live-Erkennung (OpenCV)
python3 -m src.main record     # Aufnahme-Modus (OpenCV)
python3 -m src.main status     # Status anzeigen
python3 -m src.main pipeline   # Alles auf einmal
```

## Projektstruktur

```
oegs-sign-recognition/
├── src/
│   ├── __init__.py
│   ├── main.py              # Haupteinstiegspunkt
│   ├── webapp.py            # Flask Web-Interface
│   ├── config.py            # Konfiguration
│   ├── scraper.py           # Video-Scraper für gebaerden-archiv.at
│   ├── landmark_extractor.py # MediaPipe Landmark-Extraktion
│   ├── trainer.py           # Modell-Training
│   ├── recognizer.py        # Live-Erkennung (OpenCV)
│   └── recorder.py          # Aufnahme-Modus (OpenCV)
├── templates/               # HTML-Templates für Web-Interface
│   ├── base.html
│   ├── index.html           # Dashboard
│   ├── recognize.html       # Live-Erkennung
│   └── record.html          # Aufnahme-Modus
├── data/
│   ├── videos/              # Heruntergeladene Videos
│   ├── recordings/          # Eigene Aufnahmen
│   └── landmarks/           # Extrahierte Landmarks
├── models/                  # Trainierte Modelle
├── gebaerden.txt            # Liste der akzeptierten Gebärden
├── requirements.txt
└── README.md
```

## Abhängigkeiten

- Python 3.10+
- Flask
- OpenCV
- MediaPipe
- scikit-learn
- BeautifulSoup4
- requests

## Lizenz

MIT
