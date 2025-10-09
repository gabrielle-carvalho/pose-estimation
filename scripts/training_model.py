import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os

csv_path = '../models/keypoints_dataset.csv'
model_dir = '../models/'

print("Carregando dataset")
df = pd.read_csv(csv_path)

df.dropna(inplace=True) # Remover linhas com valores ausentes

print("Preparando os dados...")
# X são as coordenadas (features), Y é a categoria da pose (target)
X = df.drop(['filename', 'pose_category'], axis=1)
y = df['pose_category']

label_encoder = LabelEncoder() # Codificar as labels (ex: 'stop' -> 1, 'nao_acao' -> 0)
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split( # 3. Dividir em treino e teste
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("Treinando o modelo RandomForest...")
model = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
model.fit(X_train, y_train)

print("Avaliando o modelo...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAcurácia no conjunto de teste: {accuracy * 100:.2f}%")
print("\nRelatório de Classificação:")
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)
print(classification_report(y_test_labels, y_pred_labels))

print("Salvando o modelo e o LabelEncoder...")
model_path = os.path.join(model_dir, 'pose_classifier_model.joblib')
encoder_path = os.path.join(model_dir, 'label_encoder.joblib')

joblib.dump(model, model_path)
joblib.dump(label_encoder, encoder_path)

print(f"\nModelo salvo em: {model_path}")
print(f"LabelEncoder salvo em: {encoder_path}")

