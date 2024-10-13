import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Crear DataFrame con datos simulados
data = pd.DataFrame({
    'fecha': pd.date_range(start='2024-01-01', periods=20),
    'hora_pico': np.random.randint(0, 2, 20),
    'dia_semana': np.random.choice(['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes'], 20),
    'num_pasajeros': np.random.randint(20, 300, 20),
    'demanda_esperada': np.random.choice(['Alta', 'Media', 'Baja'], 20),
    'clima': np.random.choice(['Soleado', 'Lluvioso', 'Nublado'], 20),
    'accidentes': np.random.randint(0, 5, 20)
})

data.to_json('transporte_masivo.json', orient='records', date_format='iso', indent=2)
data = pd.read_json('transporte_masivo.json')


# Preprocesamiento y codificación de datos
data_encoded = pd.get_dummies(data, columns=['dia_semana', 'demanda_esperada', 'clima'])
features = data_encoded[['hora_pico', 'num_pasajeros', 'accidentes']]

# Aplicar K-Means
kmeans = KMeans(n_clusters=3, random_state=42).fit(features)
data['Cluster'] = kmeans.labels_

# Crear el gráfico
plt.scatter(data['num_pasajeros'], data['accidentes'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Número de Pasajeros')
plt.ylabel('Número de Accidentes')
plt.title('Clusters de Transporte Masivo')

# Guardar el gráfico como PNG
plt.savefig('clusters_transporte.png')
print("Gráfico guardado como 'clusters_transporte.png'")
