# Structure du projet

- models/ → architectures (CNN, LSTM, MLP)
- utils/ → data.py, evaluate.py, plots.py, train.py
- main.py → fichier principal



# Installation des dépendances liées au projet 

pip install -r requirements.txt



# Lancer le projet

python main.py



# Résultats

Les résultats sont générés automatiquement dans :

- results/graphs/ → Matrices de confusion, courbes d’apprentissage, ROC
- results/metrics/ → métriques (accuracy, F1-score, etc.)
- results/reports/ → rapports de classification



# Reproductibilité

Les expériences sont exécutées 5 fois afin de garantir la robustesse des résultats.

Les métriques présentées correspondent à la moyenne et à l’écart-type.