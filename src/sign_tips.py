"""Tips for distinguishing similar ÖGS fingerspelling signs."""

# For each pair of signs that are easily confused,
# describe what to change if the user meant the OTHER sign.
# Key: (predicted, alternative) -> tip for making the alternative
_PAIR_TIPS: dict[tuple[str, str], str] = {
    # A vs S
    ("A", "S"): "Daumen seitlich anlegen (nicht vorne)",
    ("S", "A"): "Daumen vor die Finger legen",
    # A vs E
    ("A", "E"): "Finger staerker beugen, Daumen anlegen",
    ("E", "A"): "Faust bilden, Daumen vorne",
    # A vs T
    ("A", "T"): "Daumen zwischen Zeige- und Mittelfinger",
    ("T", "A"): "Daumen vor die Faust legen",
    # B vs S
    ("B", "S"): "Alle Finger zur Faust schliessen",
    ("S", "B"): "Alle Finger strecken, Hanfl. zeigen",
    # B vs D
    ("B", "D"): "Nur Zeigefinger strecken, Rest beugen",
    ("D", "B"): "Alle Finger strecken",
    # C vs O
    ("C", "O"): "Finger+Daumen zum Kreis schliessen",
    ("O", "C"): "Finger+Daumen oeffnen (C-Form)",
    # D vs F
    ("D", "F"): "Zeigefinger+Daumen zusammen, Rest strecken",
    ("F", "D"): "Nur Zeigefinger strecken",
    # E vs S
    ("E", "S"): "Faust staerker schliessen",
    ("S", "E"): "Finger leicht oeffnen, Daumen anlegen",
    # G vs H
    ("G", "H"): "Zeige+Mittelfinger strecken (seitlich)",
    ("H", "G"): "Nur Zeigefinger seitlich zeigen",
    # G vs Q
    ("G", "Q"): "Hand nach unten drehen",
    ("Q", "G"): "Hand seitlich/nach vorne zeigen",
    # H vs U
    ("H", "U"): "Hand nach oben drehen (Finger hoch)",
    ("U", "H"): "Hand seitlich drehen (Finger seitlich)",
    # I vs J
    ("I", "J"): "Kleinen Finger bewegen (J-Form zeichnen)",
    ("J", "I"): "Kleinen Finger still halten",
    # K vs V
    ("K", "V"): "Finger schliessen (V-Form)",
    ("V", "K"): "Mittelfinger + Zeigefinger spreizen",
    # L vs J
    ("L", "J"): "Kleinen Finger statt Zeigefinger",
    ("J", "L"): "Zeigefinger+Daumen strecken (L-Form)",
    # M vs N
    ("M", "N"): "Nur 2 Finger ueber dem Daumen",
    ("N", "M"): "3 Finger ueber dem Daumen",
    # P vs Q
    ("P", "Q"): "Hand nach unten drehen",
    ("Q", "P"): "Hand nach vorne zeigen",
    # R vs U
    ("R", "U"): "Zeige+Mittelfinger parallel (nicht kreuzen)",
    ("U", "R"): "Zeige+Mittelfinger kreuzen",
    # U vs V
    ("U", "V"): "Finger mehr spreizen (V-Form)",
    ("V", "U"): "Finger zusammen halten",
    # W vs V
    ("W", "V"): "Nur 2 Finger strecken (nicht 3)",
    ("V", "W"): "3 Finger strecken (Zeige+Mittel+Ring)",
    # X vs D
    ("X", "D"): "Zeigefinger strecken (nicht beugen)",
    ("D", "X"): "Zeigefinger beugen (Haken)",
    # Y vs I
    ("Y", "I"): "Nur kleinen Finger strecken",
    ("I", "Y"): "Kleiner Finger + Daumen strecken",
}


def get_tips_for_prediction(
    predicted: str,
    alternatives: list[tuple[str, float]],
    max_tips: int = 3,
) -> list[str]:
    """Get correction tips for similar alternatives.

    Args:
        predicted: The sign that was predicted.
        alternatives: List of (sign_name, confidence) for alternative signs.
        max_tips: Maximum number of tips to return.

    Returns:
        List of tip strings like "Falls du V meintest: Finger mehr spreizen"
    """
    tips = []
    for alt_name, alt_conf in alternatives:
        if alt_name == predicted or alt_name == "unknown":
            continue
        if alt_conf < 0.05:
            continue

        key = (predicted, alt_name)
        if key in _PAIR_TIPS:
            tip_text = _PAIR_TIPS[key]
        else:
            tip_text = "Handform anpassen"

        tips.append(f"Falls {alt_name}: {tip_text} ({alt_conf:.0%})")

        if len(tips) >= max_tips:
            break

    return tips
