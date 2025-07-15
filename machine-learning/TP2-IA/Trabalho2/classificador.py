import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE  
PASTA_DADOS = "dados-processados-final"

os.makedirs(PASTA_DADOS, exist_ok=True)

MODEL_FILE = os.path.join(PASTA_DADOS, "modelo_final.pkl")
ENCODER_FILE = os.path.join(PASTA_DADOS, "encoder_final.pkl")

CSV_FILE = "pre-processamento/amazon_prime_movies.csv"

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2') #modelo de embeddings utilizado 


def carregar_ou_processar_dados():
    if all(os.path.exists(f) for f in [MODEL_FILE, ENCODER_FILE]):
        print("🔁 Carregando modelo final do cache...")
        clf = joblib.load(MODEL_FILE)
        le = joblib.load(ENCODER_FILE)
    else:
        print("🧠 Treinando novo modelo com agrupamento de classes e SMOTE...")

        try:
            df = pd.read_csv(CSV_FILE)
        except FileNotFoundError:
            print(f"ERRO: Arquivo não encontrado em '{CSV_FILE}'. Verifique o caminho.")
            return None, None

        df = df[['Plot', 'Maturity Rating']]
        df.columns = ['plot', 'rating']
        df.dropna(subset=['plot', 'rating'], inplace=True)

        print("🔄️ Agrupando classes em 'Livre para todos os públicos' 'Para a familia' e 'Adulto'...")
        mapeamento_classes = {
            'All': 'Livre para todos os publicos',
            '7+': 'Livre para todos os publicos',
            '13+': 'Infanto Juvenil',                                    #agrupamento das classes para melhorar resultados 
            '16+': 'Adulto',
            '18+': 'Adulto'
        }
        df['nova_rating'] = df['rating'].map(mapeamento_classes)
        df.dropna(subset=['nova_rating'], inplace=True)
        
        print("\nDistribuição das novas classes:")
        print(df['nova_rating'].value_counts())

                                                                
        le = LabelEncoder()
        y = le.fit_transform(df['nova_rating'])         # codificar os novos rótulos 

        print(f"\n🔎 Gerando embeddings para {len(df)} sinopses...")
        embeddings = model.encode(df['plot'].tolist(), show_progress_bar=True)

       
        X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.3, random_state=42, stratify=y)  # dividir dados para treino e teste

        print(f"\n⚖️  Aplicando SMOTE para balancear as classes no conjunto de treino...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        print("⚙️  Treinando modelo LGBMClassifier com dados balanceados...")
        clf = LGBMClassifier(random_state=42) 
        clf.fit(X_train_resampled, y_train_resampled)
        print("✅ Treinamento concluído!")

        print("\n📊 Relatório de Classificação Final:")
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True,
                    xticklabels=le.classes_, yticklabels=le.classes_,
                    fmt="d", cmap="Blues")
        plt.xlabel("Predito")
        plt.ylabel("Real")
        plt.title("Matriz de Confusão Final (com SMOTE)")                       # gerando as imagens 
        plt.tight_layout()
        plt.savefig("matriz_confusao_final.png")
        print("\n🖼️ Matriz de confusão salva como 'matriz_confusao_final.png'")

        print("\n💾 Salvando modelo e encoder finais...")
        joblib.dump(clf, MODEL_FILE)
        joblib.dump(le, ENCODER_FILE)                           #salvando os resultados para evitar perda de tempo 
        print("✅ Processo concluído!")

    return clf, le


def prever_faixa_etaria(sinopse, clf, le):
    emb = model.encode([sinopse])
    pred = clf.predict(emb)
    return le.inverse_transform(pred)[0]


if __name__ == "__main__":
    clf, le = carregar_ou_processar_dados()

    if clf and le:
        while True:
            sinopse = input("\n📝 Digite uma sinopse para prever a faixa etária (ou 'sair'): ").strip()
            if sinopse.lower() == "sair":
                break
            if sinopse:
                predicao = prever_faixa_etaria(sinopse, clf, le)
                print(f"🎯 Faixa etária prevista: {predicao}")
            else:
                print("Por favor, digite uma sinopse.")