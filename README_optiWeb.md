# P8 -- MLOPS P2/2

J'ai opté pour recupérer mon EDA et apply_eda du P6, et reconstruis tout le reste a partir de ça.
Ici l'approche se focus sur la simplicité : le code est en notebook logique et tout est tracké dans mlflow.

### rappel des commandes 
(poetry)
mlflow ui --port 5000
uvicorn backend.main:app --reload
streamlit run app.py



## process suivi :

### app locale

1. #### realiser une EDA 
===> app.EDA.ipynb

2. #### creer une fonction de jointure et apply_eda()
===> app.features.optiweb.py

3. #### Réaliser une etude a travers mlflow
(puisque dans P6 je l'avais fais exclusivement en fonction, ce coup ci je redécompose tout le process dans des notebook : c'est plus clair et maintenable)
==> optiweb_pipelineTest.ipynb
  {comparaison de 3 modeles compatible GPU pour déterminé sur qui on part}
::: mlflow ui --port 5000

4. #### Réaliser une optimisation des hyperparamètres
===> optiweb_params.ipynb 
  {experience qui ameliore les perf du model cible grace a optuna (bridé a 30 min donc gain relatif 😶‍🌫️)}

5. #### Réaliser une étude pour le topk favorable
  ( On doit build un front pour exposé le model : si on utilise les 700 features on est dans la sauce !)
    {on prend le modele de reference et on entraine des modeles avec des n topk differents ( issues de feature importance) pour les comparer}
      (conclusion intéressante : avec 40features on a un bon med et 20 un light acceptable : mais a déterminer par knowledge métier.)
===> optiweb_topk.ipynb


6. #### Réaliser un export feature_meta.json 
  (avoir un dictionnaire qui permette de remplir le front de maniere coherentes à ce qu'attends le modele)
===> build_feature_web.ipynb

7. #### Réaliser une api qui permette d'exposer apply_model/ predict/ et health/  
  (pour pouvoir appeler les model depuis streamlit)
    {on appllique _select_model_and_features() directement dans predict on re sanitize rapidement et hop, ça fait des chocapics.}
    {on peut faire cela car on a ajouté un lazy reload qui detecte quel model on a }
===> backend/main.py
::: uvicorn backend.main:app --reload

8. #### Réaliser un front Streamlit  
  {expose les deux model et construisent les input en fonction de X_train}
===> frontend/app.py
::: streamlit run app.py

9. #### faire les tests 
  (fichier de test pour l'api, on va en profiter pour anticiper les log pg, en ajoutant une fonction dédié que l'on va tester : on aura plus qu'a plug une fois le container docker connecté)
  {vérifi la recup model et des appel de base, nécessite backend et mlflow Running pour être cohérent.}
===> tests/tes_api.py


### dockerisation

On a un .dockerignore qui permet de pas prendre tout le dossier.
On set via un dockerfile ou build l'image et ou on re-root touts les artefact ( chemin windows vers chemin linux)
On a le compose pour les container séparé.
l'idée :
-db pg sur 5432
-mlfloiw sur 5000
-backend sur 8000
-frontend sur 8501

=> tout run complétement en vm.


**build up**
docker compose build 
docker compose up
===> vérification log en base :
copmmande a exec dans le container de la base pg: ou avec "docker exec -it optiweb_db"
**mode sql**
psql -U optiweb -d optiweb 

**forme de la table**
prediction_logs;  

**selection**
SELECT * FROM prediction_logs ORDER BY created_at DESC LIMIT 10;
=> ça fait beacoup donc juste selet id pour voir combien de ligne



### data-drift

dossier datadrif
analyse du drift avec l'utilisation du framework evidently
 (notebook qui compare tout X_train a X_test + version light avec un top 10 )
===> drift_evidently_067.ipynb

ps: je vais ajouter un rapport avec le top 40 du topk.


===> 🤯 on a frolé la catastrophe ^^ 
Non je riogole : il faut mettre les dossier dans la root pour que ca passe bien (jai debug pendant une heur pour comprendre ahahaha) (./input vs ../input ❤️, la classique)
