FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Outils de build (pour scikit-learn, lightgbm, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# On copie seulement la conf Poetry pour profiter du cache Docker
COPY pyproject.toml poetry.lock* ./

# On installe toutes les dépendances définies dans pyproject.toml
# sans essayer de builder le projet comme un package
RUN pip install --upgrade pip \
    && pip install "poetry==1.8.3" \
    && poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi

# Puis seulement maintenant, on copie le code de l’app + mlruns + feature_meta
COPY . .

# IMPORTANT : on réécrit les chemins Windows -> chemins du container  🤯 étape qui a cassé mon crane aha
# 
RUN if [ -d "mlruns" ]; then \
      find mlruns -name "meta.yaml" -print0 | xargs -0 sed -i \
        -e 's#file:///c:/Users/suean/OneDrive/Desktop/tom/OPCL2/P8/mlruns#file:///app/mlruns#g' \
        -e 's#c:/Users/suean/OneDrive/Desktop/tom/OPCL2/P8/mlruns#/app/mlruns#g'; \
    fi

# Purement documentaire, mais tu peux le garder
EXPOSE 8000 8501 5000

# Pas de CMD ici : chaque service (backend / frontend / mlflow)
# définira sa propre commande dans docker-compose.yml
