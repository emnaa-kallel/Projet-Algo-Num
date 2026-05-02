# Projet d’Analyse Numérique  
## Interpolation et Intégration Numérique pour l’Analyse de Données Expérimentales
## Description
Ce projet vise à implémenter et analyser différentes méthodes numériques d’interpolation et d’intégration appliquées à des données expérimentales.

Deux problèmes sont étudiés :
- Le refroidissement d’un composant électronique  
- L’écoulement d’un fluide dans un canal  

L’objectif est de reconstruire une fonction continue à partir de données discrètes et de calculer des grandeurs physiques telles que la chaleur dissipée et le débit volumique.

## Objectifs
- Implémenter les méthodes d’interpolation polynomiale (Lagrange et Newton)  
- Implémenter les splines linéaires et cubiques  
- Implémenter les méthodes d’intégration numérique (rectangle, trapèzes, Simpson)  
- Implémenter une méthode d’intégration adaptative  
- Appliquer ces méthodes à des problèmes concrets  
- Analyser la précision, la stabilité et les performances  
- Résoudre un problème inverse d’estimation de paramètre  

## Structure du projet
```
TP_Analyse_Numerique/
│
├── main.py
├── interpolation/
│ ├── polynomial.py
│ ├── spline.py
│
├── integration/
│ ├── newton_cotes.py
│ ├── adaptive.py
│
├── problems/
│ ├── cooling.py
│ ├── flow.py
│
├── visualization/
│ ├── visualizer.py
│
├── data/
│ ├── cooling_data.csv
│ ├── flow_data.csv
│
├── requirements.txt
└── README.md
```
## Exécution
python main.py
## Fonctionnalités
Interpolation de données expérimentales
Calcul de températures intermédiaires
Calcul de la chaleur dissipée
Calcul du débit volumique
Étude du phénomène de Runge
Analyse de convergence des méthodes d’intégration
Visualisation graphique des résultats

## Technologies utilisées
Python 3
NumPy
Matplotlib

## Livrables
Code source structuré et documenté
Rapport (8 à 10 pages)
Résultats expérimentaux et graphiques


## Remarques
Code conforme aux conventions PEP 8
Fonctions documentées avec docstrings
Architecture modulaire et claire
## Installation
```bash
git clone https://github.com/MAHA7771/TP_Kafka.git
cd TP_Kafka
pip install -r requirements.txt


