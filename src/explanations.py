"""User-facing, version-specific explanations for the fictional teaching scenario.

Keeping the learning content separate from the statistical engine makes it
reviewable, testable and translatable without changing the calculations.
"""
from __future__ import annotations

VERSION_GUIDE = {
    "V0": {
        "question": "Wie sicher ist der anfängliche Hinweis auf einen Nutzen?",
        "observation": (
            "Vier kleine Studien liefern eine negative gepoolte Mittelwertdifferenz. "
            "Das 95%-Konfidenzintervall umfasst jedoch die Null: Ein Null-Effekt "
            "ist mit diesen Daten und dem gewählten Modell vereinbar."
        ),
        "try_this": (
            "Betrachte die einzelnen Konfidenzintervalle im Forest Plot. "
            "Warum ist das Gesamtergebnis trotz überwiegend negativer Schätzungen unsicher?"
        ),
    },
    "V1": {
        "question": "Was bewirkt die Integration einer grossen neuen Studie?",
        "observation": (
            "Gupta und SOMNI-TRIAL werden neu aufgenommen (orange markiert). "
            "Die gepoolte Schätzung verschiebt sich stärker zugunsten der Intervention; "
            "das Intervall umfasst die Null nun nicht mehr."
        ),
        "try_this": (
            "Fahre im Forest Plot mit dem Mauszeiger über SOMNI-TRIAL und lies ihr "
            "Random-Effects-Gewicht ab. Eine grosse Stichprobe bedeutet bei "
            "Heterogenität nicht automatisch ein dominierendes Gewicht."
        ),
    },
    "V2": {
        "question": "Was passiert, wenn neue grosse Studien widersprechen?",
        "observation": (
            "Drei grosse RCTs berichten kleine oder gegenteilige Effekte. "
            "Der gepoolte Effekt bewegt sich in Richtung Null und I² nimmt stark zu. "
            "Eine grössere Datenbasis führt hier nicht zu einer eindeutigeren Aussage."
        ),
        "try_this": (
            "Vergleiche V1 und V2 in der Verlaufsgrafik. Beobachte sowohl die "
            "Lage des Schätzwerts als auch die Intervallbreite und Heterogenität."
        ),
    },
    "V3": {
        "question": "Ist mehr Evidenz automatisch zuverlässiger?",
        "observation": (
            "Vier weitere Studien werden eingeschlossen. Die durchschnittliche "
            "Wirkung fällt nochmals geringer aus; die zwischen den Studien sichtbaren "
            "Unterschiede bleiben beträchtlich."
        ),
        "try_this": (
            "Untersuche die Low-RoB-Sensitivitätsanalyse. Ändert sich die Schätzung, "
            "wenn Studien mit höherem Risk of Bias nicht berücksichtigt werden?"
        ),
    },
    "V4": {
        "question": "Kann sich ein Review ohne neue Studie verändern?",
        "observation": (
            "Es kommt keine neue Studie hinzu. Stattdessen wird bei SOMNI-TRIAL "
            "der versehentlich extrahierte Vierwochenwert durch den korrekten "
            "Achtwochenwert ersetzt. Die Meta-Analyse verändert sich trotzdem."
        ),
        "try_this": (
            "Vergleiche die Zahl der Studien in V3 und V4. Öffne dann "
            "‚Workflow & Daten‘ und suche im Ereignisprotokoll die Datenkorrektur S06."
        ),
    },
}

LAB_EXPERIMENTS = [
    (
        "A · Eine grosse Nullstudie hinzufügen",
        "Ergänze über ‚Neue synthetische Studie hinzufügen‘ eine RCT mit "
        "je **500 Personen**, **MW Int. = 12**, **MW Kontr. = 12** und **SD = 4** "
        "in beiden Gruppen. Vergleiche anschliessend die gepoolte MD und das "
        "Konfidenzintervall mit der gewählten Originalversion.",
    ),
    (
        "B · Nur die Präzision verändern",
        "Setze für dieselbe neue Studie die **Stichprobengrössen auf je 30**, "
        "ohne die Mittelwerte zu ändern. Was passiert mit der Breite ihres "
        "Konfidenzintervalls und ihrem Gewicht im Forest Plot?",
    ),
    (
        "C · Daten korrigieren",
        "Ändere in einer bestehenden Studie den **Interventionsmittelwert** "
        "und klicke auf **‚Änderungen übernehmen und neu berechnen‘**. "
        "Beobachte, ob sich die Schätzung oder die Heterogenität verändert. "
        "Mit ‚Experiment zurücksetzen‘ gelangst du zum Ausgangspunkt zurück.",
    ),
]
