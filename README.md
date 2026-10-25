# Forex V1

Projet local de recherche Forex, en Python 3.11 ou plus récent. Le périmètre prévu est
long-only, sans levier et sans connexion à un broker. Aucun ordre réel n'est envoyé.

Le projet fournit l'environnement Python, une architecture modulaire, un contrat de données
et une configuration V1 validée au chargement. Les données de marché et les algorithmes
seront ajoutés dans les étapes ultérieures après validation.

## Prérequis

- Windows et PowerShell
- Python 3.11 ou plus récent, accessible avec le lanceur `py`

## Installation avec PowerShell

Depuis la racine du projet :

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Si l'activation de l'environnement est interdite par la stratégie PowerShell, utilisez
directement son interpréteur :

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Tests et lint

Avec l'environnement activé :

```powershell
python -m pytest -q
python -m ruff check .
```

Sans activation :

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

## Répertoires

- `src/forex_v1/` : package Python
- `tests/` : tests automatisés
- `configs/v1.yaml` : paramètres V1 et références de sources historiques
- `data/raw/` : données sources locales, ignorées par Git
- `data/processed/` : données préparées locales, ignorées par Git
- `reports/` : rapports générés localement, ignorés par Git
- `docs/` : documentation du projet

Aucune donnée n'est téléchargée pendant cette étape.
