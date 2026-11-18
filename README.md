# SomniTech · Living Evidence Lab

**Interaktive deutschsprachige Streamlit-Anwendung zur Demonstration von Living Systematic Reviews und kumulativen Random-Effects-Metaanalysen.**

> **Wichtig:** Alle Studien, Daten und Publikationen sind **synthetisch**. *SomniTech* ist eine fiktive App. Die Simulation enthält bewusst zukünftige Studien aus 2027. Die Ergebnisse sind **keine klinische Evidenz** und dürfen nicht für medizinische Entscheidungen verwendet werden.

## Sofort starten

Voraussetzung: Python 3.11–3.13, Terminal (macOS/Linux/Windows) und Internet für die Erstinstallation der Pakete.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Die Anwendung öffnet sich im Browser, normalerweise unter `http://localhost:8501`. Beenden mit `Ctrl+C` im Terminal.

## Vier Bereiche

1. **Living Review:** Einführung in die fiktive SomniTech-Studiengeschichte mit PICO und Erläuterung des ISI; Version V0 bis V4 wählen. Die Seite enthält eine Forest-Plot-Lesehilfe, dynamische Leitfragen zu jeder Version, die Interpretation von MD, Konfidenzintervall und I² sowie eine Low-RoB-Sensitivitätsanalyse.
2. **Experimentierlabor:** Schritt-für-Schritt-Anleitung mit drei konkreten Übungen (grosse Nullstudie, Präzision verändern, Datenkorrektur). Gruppendaten in einer Tabelle editieren, zusätzliche hypothetische RCTs hinzufügen, Random-Effects-Metaanalyse neu berechnen, Studien als CSV exportieren. Experimentelle Werte bleiben ausschliesslich in der aktuellen Streamlit-Sitzung und verändern die Original-CSV nicht.
3. **Frequentistisch vs. Bayes:** Erläuterung der beiden Unsicherheitsbegriffe, Prior-Wahl und typische Interpretationsfehler; REML/Hartung-Knapp gegen analytisch integrierte normal-normale bayesianische Random-Effects-Metaanalyse mit verstellbaren Priors und Posteriorwahrscheinlichkeit eines illustrativ relevanten Effekts.
4. **Workflow & Daten:** Suche/Screening/Extraktion/Reanalyse/Publikation; Ereignisprotokoll und Exporte für Snapshots, Versionen und Suchereignisse.

### Kurze geführte Tour

1. Öffne **01 · Living Review** und lies zunächst die fiktive Forschungsfrage. Wähle links **V0**, dann **V1** bis **V4**. Erkunde mit dem Mauszeiger Studiengewicht und Konfidenzintervalle und beachte, was sich in V4 *ohne* neue Studie verändert.
2. Öffne **02 · Experimentierlabor**. Klappe **Drei angeleitete Experimente** auf und füge z. B. eine grosse Nullstudie hinzu; beobachte rechts den Unterschied zur Originalanalyse.
3. Öffne **03 · Frequentistisch vs. Bayes**. Variiere zuerst den Prior für den durchschnittlichen Effekt μ, dann für die Heterogenität τ. Vergleiche, was das 95%-Konfidenzintervall, das 95%-Kredibilitätsintervall und P(μ < −2) aussagen.
4. Unter **04 · Workflow & Daten** siehst du, welche Schritte den simulierten Analyseversionen zugrunde liegen.

## Daten und didaktischer Verlauf

| Version | Studien | Was passiert? |
| --- | ---: | --- |
| V0 · 2024 | 4 | Basisreview mit unsicherem Effekt |
| V1 · 2025 | 6 | Grosse positive RCT verstärkt Evidenz |
| V2 · 2026 | 9 | Drei grosse Studien mit kleinen/gegenteiligen Effekten |
| V3 · 2027 | 13 | Mehr, aber heterogene Evidenz |
| V4 · 2027 | 13 | Revision des extrahierten Achtwochenwerts von S06 ohne neue Studie |

Die `studies.csv` beinhaltet die ursprünglichen und separat gespeicherten korrigierten Werte. `src/data.py` wendet die Korrektur **erst ab V4** an. Alle historischen Versionen bleiben reproduzierbar.

### Statistik

- Primärer Endpunkt: achtwöchiger Insomnia Severity Index (ISI), niedriger = besser
- **MD**: Mittelwert Intervention minus Mittelwert Kontrolle
- **Studienvarianz**: `sd_int² / n_int + sd_ctrl² / n_ctrl`
- **Frequentistisch**: Normal-Normal-Random-Effects-Modell; REML-Schätzung von τ²; 95%-KI mit modifizierter Hartung-Knapp-Methode: `max(1, Q_RE / (k−1)) / sum(weights)` und `t(k−1)`-Quantil. I² aus dem fix gewichteten Cochran-Q.
- **Bayesianisch**: `yi | μ, τ ~ Normal(μ, vi+τ²)`, `μ ~ Normal(0, σμ²)`, `τ ~ HalfNormal(στ)`; μ wird analytisch und τ auf einem ausreichend feinen Raster numerisch integriert. Standard-Priors σμ = 5 und στ = 2 ISI-Punkte; Prior-Sensitivität über die Oberfläche.
- **Illustrative Relevanzschwelle:** MD < −2 ISI-Punkte. **Keine validierte MCID-Schwelle.**

Die Analyse-Implementierung (`src/analysis.py`) ist offen, testbar und unabhängig von Streamlit. Die Tests sichern die Ergebnisse gegen feste Referenzwerte. Eine externe Kreuzvalidierung gegen `metafor::rma(..., method="REML", test="adhoc")` ist für wissenschaftlichen Einsatz zusätzlich empfehlenswert.

**Achtung:** Wiederholte frequentistische Konfidenzintervalle über zeitlich wiederkehrende Updates sind *keine* automatisch sequenziell adjustierte Inferenz. Die Demo ist kein formales Trial-Sequential-Analysis-Verfahren. Auch bayesianisches Updating beseitigt keinen Publikations- oder Selektionsbias.

## Projektstruktur

```text
app.py                   Streamlit-Oberfläche
src/analysis.py          Frequentistische und bayesianische Statistik
src/data.py              Snapshot- und Datenlogik
src/plots.py             Interaktive Plotly-Abbildungen
src/explanations.py      Didaktische Texte zu Versionen und Laborübungen
data/studies.csv         13 synthetische Studien + S06-Korrektur
data/events.csv          Ereignisprotokoll der Living-Review-Updates
tests/test_analysis.py   Regressions-, Revisions- und Plausibilitätstests
requirements.txt         Laufzeitabhängigkeiten
requirements-dev.txt     Testabhängigkeiten
.streamlit/config.toml   Darstellung
```

Tests ausführen:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## In Streamlit Community Cloud bereitstellen

1. Repository auf GitHub erstellen und diese Dateien auf den Default-Branch übertragen.
2. Auf [share.streamlit.io](https://share.streamlit.io) mit GitHub anmelden.
3. **Deploy an app**: Repository auswählen, Branch `main`, Main file path `app.py`.
4. Deploy. Es sind keine Secrets nötig.

Für die unkomplizierte öffentliche Demonstration eignet sich ein öffentliches Repository. Bei privaten Repositories hängen Freigabe und Zugriff von den GitHub-/Streamlit-Einstellungen ab.

## Lizenz

MIT. Didaktisches, rein synthetisches Beispiel.

## GitHub-Upload mit einem Befehl (optional)

Wenn GitHub CLI (`gh`) auf dem Rechner installiert und mittels `gh auth login` verbunden ist, kann das beiliegende Skript ein **eigenes Repository** erstellen und die Dateien pushen:

```bash
bash publish_to_github.sh private
```

Für eine öffentlich sichtbare Demo stattdessen `bash publish_to_github.sh public`. Das Skript benötigt eine konfigurierte lokale Git-Identität (`git config --global user.name` und `user.email`). Es verändert **keines** der bereits bestehenden GitHub-Repositories.

Ein zusätzliches R-Skript unter `validation/metafor_crosscheck.R` kann die frequentistischen Resultate unabhängig mit `metafor` prüfen; die Ausführung dieses optionalen Skripts ist nicht Voraussetzung für Streamlit.
