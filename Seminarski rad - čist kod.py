import warnings
import pandas as pd
warnings.filterwarnings("ignore")

data = pd.read_csv("Airline_customer_satisfaction.csv")
data

print(f"Dimenzije dataseta: {data.shape}")
print(f"Broj redova: {data.shape[0]}")
print(f"Broj kolona: {data.shape[1]}")

data.info()

data.describe()

data.isnull().sum()

import matplotlib.pyplot as plt
import seaborn as sns

def plot_categoric(data, column):
    plt.figure(figsize=(10, 6))
    value_counts = data[column].value_counts()

    df = value_counts.reset_index()
    df.columns = [column, 'count']

    ax = sns.barplot(
        data=df,
        x=column,
        y='count',
        order=df[column],  
        palette=['#228B22', '#FFD700', '#FF4500'][:len(df)]
    )

    plt.title(f'Barplot - {column}', fontsize=12, fontweight='bold')
    plt.xlabel(column, fontsize=10)
    plt.ylabel('Broj pojava', fontsize=10)

    for p in ax.patches:
        ax.text(
            p.get_x() + p.get_width() / 2,
            p.get_height() + (max(value_counts.values) * 0.01),
            int(p.get_height()),
            ha='center',
            va='bottom',
            fontsize=9
        )

    plt.tight_layout()
    plt.show()

    print(f"\nDistribucija kategorija za {column}:")
    print(value_counts)
    
columns = ["satisfaction", "Customer Type", "Type of Travel","Class"]

# Analiza svih kategorijskih kolona

for column in columns:
    print(f"\n\n\n\n{'='*60}")
    print(f"{column.upper()}")
    print('='*60)
    plot_categoric(data, column)
    
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np

def plot_numeric_nongrades(data, column):
    plt.figure(figsize=(14, 6))
    
    # Boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(x=data[column], color="skyblue")
    plt.title(f'Boxplot - {column}', fontsize=12, fontweight='bold')
    plt.xlabel(column, fontsize=10)
    
    # Histogram
    plt.subplot(1, 2, 2)
    sns.histplot(data[column], kde=True, color="darkgreen", bins=30)
    plt.title(f'Histogram - {column}', fontsize=12, fontweight='bold')
    plt.xlabel(column, fontsize=10)
    plt.ylabel('Frekvencija', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Osnovne statistike
    mean_val = data[column].mean()
    median_val = data[column].median()
    std_val = data[column].std()
    min_val = data[column].min()
    max_val = data[column].max()
    
    print(f"\nOsnovne statistike za {column}:")
    print(f"  Srednja vrednost: {mean_val:.2f}")
    print(f"  Medijana: {median_val:.2f}")
    print(f"  St. devijacija: {std_val:.2f}")
    print(f"  Min: {min_val:.2f}")
    print(f"  Max: {max_val:.2f}")
    
    # D'Agostino K^2 test za normalnost
    stat, p = stats.normaltest(data[column])
    print(f"\nD'Agostino K² test za normalnost:")
    print(f"  Statistika = {stat:.3f}, p-vrednost = {p:.3f}")
    if p > 0.05:
        print("Podaci su priblizno normalno rasporedjeni (p > 0.05).")
    else:
        print("Podaci nisu normalno rasporedjeni (p < 0.05).")
    
    # Detekcija outliera (IQR)
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower) | (data[column] > upper)]
    outlier_pct = len(outliers) / len(data) * 100
    
    print(f"\nDetekcija outliera (IQR metoda):")
    print(f"  Donja granica: {lower:.2f}, Gornja granica: {upper:.2f}")
    print(f"  Broj outliera: {len(outliers)} ({outlier_pct:.2f}%)")
    
    if outlier_pct > 5:
        print("\n -Postoji znacajan broj outliera — distribucija je asimetricna i ima dugacak rep.")
    else:
        print("\n -Outlieri su retki i ne uticu znacajno na raspodelu.")


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_numeric_grades(data, column):
    value_counts_sorted = data[column].value_counts().sort_index()
    
    plt.figure(figsize=(10, 6))
    
    palette = sns.color_palette("RdYlGn", 6)

    sns.barplot(x=value_counts_sorted.index, 
                y=value_counts_sorted.values, 
                palette=palette)
    
    plt.title(f'Barplot - {column}', fontsize=12, fontweight='bold')
    plt.xlabel(column, fontsize=10)
    plt.ylabel('Broj pojava', fontsize=10)
    
    for i, v in enumerate(value_counts_sorted.values):
        plt.text(i, v + max(value_counts_sorted.values) * 0.01, 
                 str(v), 
                 ha='center', 
                 fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nDistribucija ocene za {column}:")
    print(value_counts_sorted)
    
notColumns = ["satisfaction", "Customer Type", "Type of Travel", "Class",
              "Age","Flight Distance","Departure Delay in Minutes","Arrival Delay in Minutes"]
columns = [column for column in data.columns if column not in notColumns]
# Analiza svih Numerički kolona koje predstavljaju ocene.


for column in columns:
    print(f"\n\n\n\n{'='*60}")
    print(f"{column.upper()}")
    print('='*60)
    plot_numeric_grades(data, column)
    
import numpy as np

notColumns = [
    "satisfaction", "Customer Type", "Type of Travel", "Class",
    "Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"
]


columns_to_clean = [column for column in data.columns if column not in notColumns]



for col in columns_to_clean:
    if pd.api.types.is_numeric_dtype(data[col]):
        initial_nan_count = data[col].isnull().sum()
        

        zero_count = (data[col] == 0).sum()
        
        # Izvršavanje zamene
        data[col].replace(0, np.nan, inplace=True)
        

print("\nCisenje je završeno. Novi NaN-ovi su sada spremni za imputaciju (npr. Modom ili KNN-om).")

from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler,OneHotEncoder


data_knn = data.copy()


numeric_cols = data_knn.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = data_knn.select_dtypes(include=['object']).columns.tolist()

nominal_cols = ['satisfaction', 'Customer Type', 'Type of Travel']
ordinal_cols = [col for col in categorical_cols if col not in nominal_cols]

#Enkodiranje kategoričkih promenljivih (Ovo ćemo kasnije takođe raditi pre modeliranja)
label_encoders = {}
for col in ordinal_cols:
    le = LabelEncoder()
    data_knn[col] = le.fit_transform(data_knn[col].astype(str))
    label_encoders[col] = le
    
    
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe_data = pd.DataFrame(
    ohe.fit_transform(data_knn[nominal_cols]),
    columns=ohe.get_feature_names_out(nominal_cols),
    index=data_knn.index
)    

data_knn = pd.concat([data_knn[numeric_cols + ordinal_cols], ohe_data], axis=1)

# Skaliranje numeričkih promenljivih
scaler = StandardScaler()
data_knn[numeric_cols] = scaler.fit_transform(data_knn[numeric_cols])

# Moramo na 3, ako stavimo na 5, izvršavanje će trajati minut i po!!
knn_imputer = KNNImputer(n_neighbors=3) 
data_imputed_array = knn_imputer.fit_transform(data_knn)

data_imputed = pd.DataFrame(data_imputed_array, columns=data_knn.columns)



#Inverzno skaliranje numeričkih kolona
data_imputed[numeric_cols] = scaler.inverse_transform(data_imputed[numeric_cols])


for col in ordinal_cols:
    data_imputed[col] = data_imputed[col].round().astype(int)
    data_imputed[col] = label_encoders[col].inverse_transform(data_imputed[col])


data['Arrival Delay in Minutes'] = data_imputed['Arrival Delay in Minutes']

# Provera posle imputacije
print("\nUspeh: 'Arrival Delay in Minutes' je imputirana.")
print(f"NaN u 'Arrival Delay in Minutes' posle KNN: {data['Arrival Delay in Minutes'].isnull().sum()}")

data.isnull().sum()

grade_columns = [
    'Seat comfort',                        # 4797 NULA
    'Food and drink',                      # 5945 NULA
    'Departure/Arrival time convenient',   # 6664 NULA
    'Inflight entertainment',              # 2978 NULA
    'Leg room service',                    # 444
    'Inflight wifi service',               # 132 NULA
    'Ease of Online booking',              # 18 NULA
    'Cleanliness',                         # 5 NULA
    'On-board service',                    # 5 NULA
    'Online boarding',                     # 14 NULA
    'Online support',                      # 1 NULA
    'Gate location',                       # 2 NULE
    'Checkin service',                     # 1 NULA
]

NAN_THRESHOLD = 50 
rows_dropped_total = 0

for col in grade_columns:
    nan_count = data[col].isnull().sum()
    
    # Proveravamo samo kolone koje imaju NaN vrednosti
    if nan_count > 0:
        if nan_count < NAN_THRESHOLD:
            # Brisanje (za mali broj anomaličnih 0)
            rows_before = len(data)
            data.dropna(subset=[col], inplace=True)
            rows_dropped = rows_before - len(data)
            rows_dropped_total += rows_dropped
                   
        else:
            # Imputacija Modom (za veliki broj anomaličnih 0)
            modus_vrednost = data[col].mode()[0] 
            data[col].fillna(modus_vrednost, inplace=True)
            
            data[col] = data[col].astype(int)

print(f"\nUKUPNO REDOVA OBRISANO: {rows_dropped_total}")
print("Sve kolone sa ocenama su sada ciste i sadrze samo vrednosti od 1 do 5.")
data.isnull().sum()


import warnings
import math
warnings.filterwarnings("ignore")

notColumns = ["satisfaction", "Customer Type", "Type of Travel", "Class",
              "Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
columns = [column for column in data.columns if column not in notColumns]


def plot_grade_grid(data, grade_columns, cols_per_row=3, fig_width=15, fig_height_per_row=4):
   
    num_cols = len(grade_columns)
    num_rows = math.ceil(num_cols / cols_per_row)
    
    fig, axes = plt.subplots(num_rows, cols_per_row, 
                             figsize=(fig_width, num_rows * fig_height_per_row))
    
    
    if num_rows == 1 and cols_per_row == 1:
        axes = np.array([axes])
    elif num_rows == 1 or cols_per_row == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    palette = sns.color_palette("RdYlGn", 6)

    for i, col in enumerate(grade_columns):
        ax = axes[i]
        
        value_counts_sorted = data[col].value_counts().sort_index()

        sns.barplot(x=value_counts_sorted.index, 
                    y=value_counts_sorted.values, 
                    palette=palette, 
                    ax=ax)
        
        ax.set_title(f'{col}', fontsize=12)
        ax.set_xlabel('Ocena', fontsize=10)
        ax.set_ylabel('Broj', fontsize=10)
        
        max_v = max(value_counts_sorted.values) if len(value_counts_sorted.values) > 0 else 0
        for j, v in enumerate(value_counts_sorted.values):
            ax.text(j, v + max_v * 0.01, 
                    str(v), 
                    ha='center', 
                    fontsize=8, 
                    weight='bold')

    for i in range(num_cols, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()
    
plot_grade_grid(data, columns, cols_per_row=3)

delay_cols = ["Departure Delay in Minutes","Arrival Delay in Minutes"]


k_multiplier = 3.0 

for col in delay_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    
    # Računanje gornje granice
    upper_bound = Q3 + k_multiplier * IQR
    
    data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
    
    print(f"Varijabla '{col}': Gornja granica (k={k_multiplier}) postavljena na {upper_bound:.2f}")

columns = ["Departure Delay in Minutes","Arrival Delay in Minutes"]
# Ponovni prikaz Departure i Arrival Delaya.


for column in columns:
    print(f"\n\n\n\n{'='*60}")
    print(f"{column.upper()}")
    print('='*60)
    plot_numeric_nongrades(data, column)
    

def cliffs_delta(x, y):
    n1, n2 = len(x), len(y)
    greater = sum(i > j for i in x for j in y)
    less = sum(i < j for i in x for j in y)
    delta = (greater - less) / (n1 * n2)
    return delta
def cramers_v(confusion_matrix):
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(k, r) - 1)))

ColumnsNumericNonGrades = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]


def plot_target_relationship(data, target_column, feature_column):
    #Uzorkovanje
    if len(data) > 10000:
        data = data.sample(n=10000, random_state=42)

    # NUMERIČKE promenljive
    if data[feature_column].dtype in ['int64', 'float64', 'int32']:
        #Ako su ocene (1–5) - (boxplot, histogram, heatmap)
        if feature_column not in ColumnsNumericNonGrades:
            fig, axes = plt.subplots(1, 3, figsize=(22, 6))

            # Boxplot
            sns.boxplot(
                x=target_column,
                y=feature_column,
                data=data,
                palette={'satisfied': "#008437", 'dissatisfied': "#ffa352"},
                ax=axes[0]
            )
            axes[0].set_title(f'Boxplot: {feature_column} po {target_column}', fontsize=12, fontweight='bold')

            # Histogram
            sns.histplot(
                data=data,
                x=feature_column,
                hue=target_column,
                multiple='dodge',
                shrink=0.8,
                discrete=True,
                palette={'satisfied': '#008437', 'dissatisfied': '#ffa352'},
                ax=axes[1]
            )
            axes[1].set_title(f'Histogram (1–5 skala): {feature_column}', fontsize=12, fontweight='bold')

            #Heatmap (cross-tab)
            ct = pd.crosstab(data[feature_column], data[target_column])
            sns.heatmap(ct, annot=True, fmt='d', cmap='YlGnBu', ax=axes[2])
            axes[2].set_title(f'Heatmap: {feature_column} × {target_column}', fontsize=12, fontweight='bold')

            plt.tight_layout()
            plt.show()

        #Ako su prave numeričke (Age, Delay itd.) - 2 grafa (boxplot + density)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            sns.boxplot(
                x=target_column,
                y=feature_column,
                data=data,
                palette={'satisfied': "#008437", 'dissatisfied': "#ffa352"},
                ax=axes[0]
            )
            axes[0].set_title(f'Boxplot: {feature_column} po {target_column}', fontsize=12, fontweight='bold')

            sns.kdeplot(
                data=data,
                x=feature_column,
                hue=target_column,
                fill=True,
                common_norm=False,
                palette={'satisfied': '#008437', 'dissatisfied': '#ffa352'},
                alpha=0.5,
                ax=axes[1]
            )
            axes[1].set_title(f'Density Plot: {feature_column} po {target_column}', fontsize=12, fontweight='bold')

            plt.tight_layout()
            plt.show()

        #STATISTIČKI TEST: Mann–Whitney + Cliff’s Delta
        categories = sorted(data[target_column].unique())
        if len(categories) == 2:
            group1 = data[data[target_column] == categories[0]][feature_column].dropna()
            group2 = data[data[target_column] == categories[1]][feature_column].dropna()
            if len(group1) > 0 and len(group2) > 0:
                u_stat, u_p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                delta = cliffs_delta(group1.sample(min(500, len(group1))), group2.sample(min(500, len(group2))))
                print(f"Mann-Whitney U test - p = {u_p_value:.2e}, Cliff's Delta = {delta:.3f}")
                if abs(delta) < 0.1:
                    print("Vrlo slab efekat")
                elif abs(delta) < 0.3:
                    print("Srednji efekat")
                else:
                    print("Jak efekat")
        print("-" * 80)

    # KATEGORIJSKE PROMENLJIVE 
    else:
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))

        ct_pct = pd.crosstab(data[feature_column], data[target_column], normalize='index') * 100
        ct_pct.plot(kind='bar', stacked=False, color=['#ffa352', '#008437'], ax=ax[0])
        ax[0].set_title(f'{feature_column} vs {target_column} (Procenat)', fontsize=12, fontweight='bold')
        ax[0].set_xlabel(feature_column)
        ax[0].set_ylabel('Procenat (%)')
        ax[0].legend(title=target_column)
        plt.setp(ax[0].get_xticklabels(), rotation=45)

        ct = pd.crosstab(data[feature_column], data[target_column])
        sns.heatmap(ct, annot=True, fmt='d', cmap='YlGnBu', ax=ax[1])
        ax[1].set_title(f'Cross-tab Heatmap: {feature_column} × {target_column}', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.show()

        chi2_stat, chi2_p_value, dof, expected = stats.chi2_contingency(ct)
        v = cramers_v(ct)
        if v < 0.1:
            strength = "vrlo slaba"
        elif v < 0.3:
            strength = "umerena"
        else:
            strength = "jaka"
        print(f"Chi-square test ,p = {chi2_p_value:.2e}, Cramer's V = {v:.3f} ({strength} veza)")
        print("-" * 80)

for column in data.columns:
    if column != 'satisfaction':
        print(f"\n{'-'*80}")
        print(f"ANALIZA: {column.upper()} vs SATISFACTION")
        print('-'*80)
        plot_target_relationship(data, 'satisfaction', column)
        
data.info()


numeric_cols = data.select_dtypes(include=['int64', 'float64','int32']).columns

plt.figure(figsize=(16, 14))
correlation_matrix = data[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Korelaciona matrica numerickih promenljivih', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nVisoko korelisane promenljive (|r| > 0.7):")
high_corr = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            high_corr.append({
                'Promenljiva 1': correlation_matrix.columns[i],
                'Promenljiva 2': correlation_matrix.columns[j],
                'Korelacija': correlation_matrix.iloc[i, j]
            })

if high_corr:
    high_corr_df = pd.DataFrame(high_corr)
    print(high_corr_df.to_string(index=False))
else:
    print("Nema visoko korelisanih promenljivih.")
    
    
columns_to_drop = [
    'Gate location',
    'Departure/Arrival time convenient',
]

data = data.drop(columns=columns_to_drop, errors='ignore')

service_columns_to_engineer = [
    'Seat comfort', 
    'Food and drink', 
    'Inflight wifi service', 
    'Inflight entertainment',
    'On-board service',
    'Leg room service',
    'Cleanliness',
]

existing_service_cols = [col for col in service_columns_to_engineer if col in data.columns]

if existing_service_cols:
    
    data[existing_service_cols] = data[existing_service_cols].replace(0, np.nan) 
    
    data['Flight_Experience'] = (
        data[existing_service_cols]
        .mean(axis=1)   
        .round()        
        .astype(int)   
    )

    data = data.drop(columns=existing_service_cols, errors='ignore')

print('='*60)
print(f"{'Flight_Experience'.upper()}")
print('='*60)
plot_numeric_grades(data, 'Flight_Experience')
    
print(f"\n{'-'*80}")
print(f"ANALIZA: {'Flight_Experience'.upper()} vs SATISFACTION")
print('-'*80)
plot_target_relationship(data, 'satisfaction', 'Flight_Experience')


Ground_Service_Quality = [
    'Online support', 
    'Ease of Online booking', 
    'Baggage handling', 
    'Checkin service',
    'Online boarding'
]

existing_service_cols = [col for col in Ground_Service_Quality if col in data.columns]

if existing_service_cols:
       
    data['Ground_Service_Quality'] = (
        data[existing_service_cols]
        .mean(axis=1)   
        .round()        
        .astype(int)   
    )

    data = data.drop(columns=existing_service_cols, errors='ignore')
    
    
print('='*60)
print(f"{'Ground_Service_Quality'.upper()}")
print('='*60)
plot_numeric_grades(data, 'Ground_Service_Quality')
    
print(f"\n{'-'*80}")
print(f"ANALIZA: {'Ground_Service_Quality'.upper()} vs SATISFACTION")
print('-'*80)
plot_target_relationship(data, 'satisfaction', 'Ground_Service_Quality')

service_columns_to_engineer = [
    'Departure Delay in Minutes', 
    'Arrival Delay in Minutes', 
]

existing_service_cols = [col for col in service_columns_to_engineer if col in data.columns]

if existing_service_cols:
      
    data['Total_Delay_in_minutes'] = (
        data[existing_service_cols[0]] + data[existing_service_cols[1]] 
    )

    data = data.drop(columns=existing_service_cols, errors='ignore')
    
ColumnsNumericNonGrades.append('Total_Delay_in_minutes')

print('='*60)
print(f"{'Total_Delay_in_minutes'.upper()}")
print('='*60)
plot_numeric_nongrades(data, 'Total_Delay_in_minutes')
    
print(f"\n{'-'*80}")
print(f"ANALIZA: {'Total_Delay_in_minutes'.upper()} vs SATISFACTION")
print('-'*80)
plot_target_relationship(data, 'satisfaction', 'Total_Delay_in_minutes')


max_age = data['Age'].max() if 'Age' in data.columns and not data['Age'].empty else 100
    
bins = [0, 30, 55, max_age + 1] 
    
labels = ['Young', 'Adult', 'Senior']
    
data['Age_Group'] = pd.cut(
    data['Age'], 
    bins=bins, 
    labels=labels, 
    right=False, 
    include_lowest=True, # Uključuje najnižu vrednost (0)
)

print("Kolona 'Age' je uspešno kategorizovana u 'Age_Group'.")
print(f"Kategorije: {labels}")


print('='*60)
print(f"{'Age_Group'.upper()}")
print('='*60)
plot_categoric(data, 'Age_Group')
    
print(f"\n{'-'*80}")
print(f"ANALIZA: {'Age_Group'.upper()} vs SATISFACTION")
print('-'*80)
plot_target_relationship(data, 'satisfaction', 'Age_Group')


max_dist = data['Flight Distance'].max() if 'Flight Distance' in data.columns and not data['Flight Distance'].empty else 7000
    
bins = [0, 1000, 4000, max_dist + 1] 
    
labels = ['Short', 'Medium', 'Long']
    
data['Flight_Distance_Group'] = pd.cut(
    data['Flight Distance'], 
    bins=bins, 
    labels=labels, 
    right=False, 
    include_lowest=True, 
)

print("Kolona 'Flight Distance' je uspešno kategorizovana u 'Flight_Distance_Group'.")
print(f"Kategorije: {labels}")

print('='*60)
print(f"{'Flight_Distance_Group'.upper()}")
print('='*60)
plot_categoric(data, 'Flight_Distance_Group')
    
print(f"\n{'-'*80}")
print(f"ANALIZA: {'Flight_Distance_Group'.upper()} vs SATISFACTION")
print('-'*80)
plot_target_relationship(data, 'satisfaction', 'Flight_Distance_Group')

max_dist = data['Total_Delay_in_minutes'].max() if 'Total_Delay_in_minutes' in data.columns and not data['Total_Delay_in_minutes'].empty else 100

bins = [0, 3, 15, 60, max_dist + 1]    

labels = ['No_Delay','Minor_Delay', 'Medium_Delay', 'Critical_Delay']

data['Total_Delay_in_minutes_group'] = pd.cut(
    data['Total_Delay_in_minutes'],
    bins=bins,
    labels=labels,
    right=False,
    include_lowest=True,
)

print("Kolona 'Total_Delay_in_minutes' je uspešno kategorizovana u 'Total_Delay_in_minutes_group'.")
print(f"Kategorije: {labels}")

print('='*60)
print(f"{'Total_Delay_in_minutes_group'.upper()}")
print('='*60)
plot_categoric(data, 'Total_Delay_in_minutes_group')
    
print(f"\n{'-'*80}")
print(f"ANALIZA: {'Total_Delay_in_minutes_group'.upper()} vs SATISFACTION")
print('-'*80)
plot_target_relationship(data, 'satisfaction', 'Total_Delay_in_minutes_group')

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, OrdinalEncoder

TARGET_COL = 'satisfaction'

le = LabelEncoder()
data['Satisfaction_encoded'] = le.fit_transform(data['satisfaction'])
print(data[['satisfaction', 'Satisfaction_encoded']])

#Ostavljamo originalnu Satisfaction, radi lakšeg čitanja kasnije

# Kontinuirane numeričke (za Standard Scaling)
SCALE_COLS = ['Age', 'Flight Distance', 'Total_Delay_in_minutes']

scaler = StandardScaler()

data[SCALE_COLS] = scaler.fit_transform(data[SCALE_COLS])

OHE_COLS = ['Type of Travel', 'Customer Type']

data = data.reset_index(drop=True) 

ohe = OneHotEncoder(drop='first', sparse_output=False) 

encoded_array = ohe.fit_transform(data[OHE_COLS])

encoded_data = pd.DataFrame(
    encoded_array, 
    columns=ohe.get_feature_names_out(OHE_COLS),
    index=data.index 
)


data = pd.concat([data.drop(OHE_COLS, axis=1), encoded_data], axis=1)
print("\nUspeh: One-Hot Encoding je izvršen bez kreiranja novih NaN vrednosti.")

# Ordinalne/Grade/Grupne (za Ordinal Encoding)
ORDINAL_COLS = [
    'Class', 
    'Age_Group', 
    'Flight_Distance_Group', 
    'Total_Delay_in_minutes_group'
]

categories = [
    ['Eco', 'Eco Plus', 'Business'],                        
    ['Young', 'Adult', 'Senior'], 
    ['Short', 'Medium', 'Long'],             
    ['No_Delay', 'Minor_Delay', 'Medium_Delay', 'Critical_Delay']           
]
print(data.isnull().sum())


ordinal_encoder = OrdinalEncoder(categories=categories)
data[ORDINAL_COLS] = ordinal_encoder.fit_transform(data[ORDINAL_COLS])

data.info()
data
data.to_csv("clean_airline_data.csv", index=False)




# Import biblioteka za modelovanje
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Import metrika za evaluaciju
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc
)


def evaluate_model_cv(model, X, y, model_name, n_splits=5, random_state=42):
    """
    Evaluira model korišćenjem stratified k-fold cross-validacije.
    """
    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    metrics_list = []
    
    print(f"\n{'='*80}")
    print(f"EVALUACIJA MODELA: {model_name}")
    print('='*80)
    
    for fold, (train_idx, val_idx) in enumerate(stratified_kfold.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Treniranje modela
        model.fit(X_train, y_train)
        
        # Predikcije
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Metrike za trening set
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
        train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
        
        # Metrike za validacioni set
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
        val_recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
        
        metrics_list.append({
            'Fold': fold,
            'Train Accuracy': train_accuracy,
            'Train Precision': train_precision,
            'Train Recall': train_recall,
            'Train F1': train_f1,
            'Val Accuracy': val_accuracy,
            'Val Precision': val_precision,
            'Val Recall': val_recall,
            'Val F1': val_f1
        })
        
        print(f"\nFold {fold}:")
        print(f"  Train - Acc: {train_accuracy:.4f}, Prec: {train_precision:.4f}, Rec: {train_recall:.4f}, F1: {train_f1:.4f}")
        print(f"  Val   - Acc: {val_accuracy:.4f}, Prec: {val_precision:.4f}, Rec: {val_recall:.4f}, F1: {val_f1:.4f}")
    
    # Kreiranje DataFrame-a sa metrikama
    metrics_df = pd.DataFrame(metrics_list)
    
    # Prosečne metrike
    print(f"\n{'='*80}")
    print("PROSECNE METRIKE:")
    print('='*80)
    print(f"Train - Acc: {metrics_df['Train Accuracy'].mean():.4f} (±{metrics_df['Train Accuracy'].std():.4f})")
    print(f"        Prec: {metrics_df['Train Precision'].mean():.4f} (±{metrics_df['Train Precision'].std():.4f})")
    print(f"        Rec: {metrics_df['Train Recall'].mean():.4f} (±{metrics_df['Train Recall'].std():.4f})")
    print(f"        F1: {metrics_df['Train F1'].mean():.4f} (±{metrics_df['Train F1'].std():.4f})")
    print(f"\nVal   - Acc: {metrics_df['Val Accuracy'].mean():.4f} (±{metrics_df['Val Accuracy'].std():.4f})")
    print(f"        Prec: {metrics_df['Val Precision'].mean():.4f} (±{metrics_df['Val Precision'].std():.4f})")
    print(f"        Rec: {metrics_df['Val Recall'].mean():.4f} (±{metrics_df['Val Recall'].std():.4f})")
    print(f"        F1: {metrics_df['Val F1'].mean():.4f} (±{metrics_df['Val F1'].std():.4f})")
    
    return metrics_df

print("Funkcija za evaluaciju modela kreirana!")

# Priprema podataka za modelovanje
X_data = data.drop(columns=['satisfaction', 'Satisfaction_encoded'], errors='ignore')
y_data = data['Satisfaction_encoded']

print(f"Dimenzije X: {X_data.shape}")
print(f"Dimenzije y: {y_data.shape}")
print(f"\nDistribucija ciljne promenljive:")
print(y_data.value_counts())

# Treniranje Logistic Regression modela
logreg_model = LogisticRegression(max_iter=1000, random_state=42)

# Evaluacija modela
logreg_metrics = evaluate_model_cv(logreg_model, X_data, y_data, "Logistic Regression")

# Grid Search za optimalne hiperparametre Decision Tree-a
from sklearn.model_selection import GridSearchCV


param_grid_dt = {
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

dt_model = DecisionTreeClassifier(random_state=42)
grid_search_dt = GridSearchCV(dt_model, param_grid_dt, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

print("Pokretanje Grid Search-a za Decision Tree...")
grid_search_dt.fit(X_data, y_data)

print(f"\nNajbolji parametri: {grid_search_dt.best_params_}")
print(f"Najbolji score: {grid_search_dt.best_score_:.4f}")

# Evaluacija Decision Tree modela sa optimalnim parametrima
best_dt_model = DecisionTreeClassifier(**grid_search_dt.best_params_, random_state=42)
dt_metrics = evaluate_model_cv(best_dt_model, X_data, y_data, "Decision Tree (Optimized)")

# Randomized Search za Random Forest (brže od Grid Search-a)
param_dist_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf_model = RandomForestClassifier(random_state=42)
random_search_rf = RandomizedSearchCV(rf_model, param_dist_rf, n_iter=20, cv=5, 
                                       scoring='accuracy', n_jobs=-1, verbose=1, random_state=42)

print("Pokretanje Randomized Search-a za Random Forest...")
random_search_rf.fit(X_data, y_data)

print(f"\nNajbolji parametri: {random_search_rf.best_params_}")
print(f"Najbolji score: {random_search_rf.best_score_:.4f}")

# Evaluacija Random Forest modela
best_rf_model = RandomForestClassifier(**random_search_rf.best_params_, random_state=42)
rf_metrics = evaluate_model_cv(best_rf_model, X_data, y_data, "Random Forest (Optimized)")

# Randomized Search za XGBoost


param_dist_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
random_search_xgb = RandomizedSearchCV(xgb_model, param_dist_xgb, n_iter=20, cv=5,
                                        scoring='accuracy', n_jobs=-1, verbose=1, random_state=42)

print("Pokretanje Randomized Search-a za XGBoost...")
random_search_xgb.fit(X_data, y_data)

print(f"\nNajbolji parametri: {random_search_xgb.best_params_}")
print(f"Najbolji score: {random_search_xgb.best_score_:.4f}")

# Evaluacija XGBoost modela
best_xgb_model = XGBClassifier(**random_search_xgb.best_params_, random_state=42, eval_metric='logloss')
xgb_metrics = evaluate_model_cv(best_xgb_model, X_data, y_data, "XGBoost (Optimized)")

# Poređenje svih modela
comparison_data = {
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost'],
    'Train Accuracy': [
        logreg_metrics['Train Accuracy'].mean(),
        dt_metrics['Train Accuracy'].mean(),
        rf_metrics['Train Accuracy'].mean(),
        xgb_metrics['Train Accuracy'].mean()
    ],
    'Val Accuracy': [
        logreg_metrics['Val Accuracy'].mean(),
        dt_metrics['Val Accuracy'].mean(),
        rf_metrics['Val Accuracy'].mean(),
        xgb_metrics['Val Accuracy'].mean()
    ],
    'Train F1': [
        logreg_metrics['Train F1'].mean(),
        dt_metrics['Train F1'].mean(),
        rf_metrics['Train F1'].mean(),
        xgb_metrics['Train F1'].mean()
    ],
    'Val F1': [
        logreg_metrics['Val F1'].mean(),
        dt_metrics['Val F1'].mean(),
        rf_metrics['Val F1'].mean(),
        xgb_metrics['Val F1'].mean()
    ]
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.round(4)

print("\n" + "="*80)
print("POREDJENJE PERFORMANSI MODELA")
print("="*80)
print(comparison_df.to_string(index=False))

# Vizualizacija poređenja
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Accuracy
x = np.arange(len(comparison_df['Model']))
width = 0.35

axes[0].bar(x - width/2, comparison_df['Train Accuracy'], width, label='Train', color='skyblue')
axes[0].bar(x + width/2, comparison_df['Val Accuracy'], width, label='Validation', color='coral')
axes[0].set_xlabel('Model', fontweight='bold')
axes[0].set_ylabel('Accuracy', fontweight='bold')
axes[0].set_title('Poređenje Accuracy-ja modela', fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(comparison_df['Model'], rotation=15)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# F1 Score
axes[1].bar(x - width/2, comparison_df['Train F1'], width, label='Train', color='lightgreen')
axes[1].bar(x + width/2, comparison_df['Val F1'], width, label='Validation', color='lightcoral')
axes[1].set_xlabel('Model', fontweight='bold')
axes[1].set_ylabel('F1 Score', fontweight='bold')
axes[1].set_title('Poređenje F1 Score-a modela', fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(comparison_df['Model'], rotation=15)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


# Kreiranje train/test split-a za finalno testiranje
X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)

X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)

# Treniranje svih modela na train setu
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(**grid_search_dt.best_params_, random_state=42),
    'Random Forest': RandomForestClassifier(**random_search_rf.best_params_, random_state=42),
    'XGBoost': XGBClassifier(**random_search_xgb.best_params_, random_state=42, eval_metric='logloss')
}

# Treniranje modela
print("Treniranje finalnih modela...")
models['Logistic Regression'].fit(X_train_linear, y_train_linear)
models['Decision Tree'].fit(X_train_tree, y_train_tree)
models['Random Forest'].fit(X_train_tree, y_train_tree)
models['XGBoost'].fit(X_train_tree, y_train_tree)

print("Svi modeli su uspesno trenirani!")

# ROC krive za sve modele
plt.figure(figsize=(12, 8))

# Kreiranje binarnih labela (1 za 'satisfied', 0 za ostalo)
# Za linear model
y_test_linear_binary = (y_test_linear == 'satisfied').astype(int) if y_test_linear.dtype == 'object' else y_test_linear

# Za tree model (provera da li je već enkodiran)
if y_test_tree.dtype == 'object':
    y_test_tree_binary = (y_test_tree == 'satisfied').astype(int)
else:
    # Ako je već enkodiran, pretpostavljamo da je 1 = 'satisfied'
    y_test_tree_binary = y_test_tree

# Logistic Regression
y_proba_lr = models['Logistic Regression'].predict_proba(X_test_linear)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test_linear_binary, y_proba_lr)
auc_lr = auc(fpr_lr, tpr_lr)
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.3f})', linewidth=2, color='blue')

# Decision Tree
y_proba_dt = models['Decision Tree'].predict_proba(X_test_tree)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_test_tree_binary, y_proba_dt)
auc_dt = auc(fpr_dt, tpr_dt)
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc_dt:.3f})', linewidth=2, color='green')

# Random Forest
y_proba_rf = models['Random Forest'].predict_proba(X_test_tree)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test_tree_binary, y_proba_rf)
auc_rf = auc(fpr_rf, tpr_rf)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.3f})', linewidth=2, color='orange')

# XGBoost
y_proba_xgb = models['XGBoost'].predict_proba(X_test_tree)[:, 1]
fpr_xgb, tpr_xgb, _ = roc_curve(y_test_tree_binary, y_proba_xgb)
auc_xgb = auc(fpr_xgb, tpr_xgb)
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.3f})', linewidth=2, color='red')

# Dijagonala (random classifier)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)

plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('ROC krive svih modela', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\nAUC skorovi:")
print(f"  Logistic Regression: {auc_lr:.4f}")
print(f"  Decision Tree: {auc_dt:.4f}")
print(f"  Random Forest: {auc_rf:.4f}")
print(f"  XGBoost: {auc_xgb:.4f}")

# Određivanje najboljeg modela (po AUC skoru)
auc_scores = {'Logistic Regression': auc_lr, 'Decision Tree': auc_dt, 
              'Random Forest': auc_rf, 'XGBoost': auc_xgb}
best_model_name = max(auc_scores, key=auc_scores.get)

print(f"\nNajbolji model: {best_model_name} (AUC = {auc_scores[best_model_name]:.4f})")

# Odabir najboljeg modela i odgovarajućeg test seta
if best_model_name == 'Logistic Regression':
    best_model = models['Logistic Regression']
    X_test_best = X_test_linear
    y_test_best = y_test_linear
else:
    best_model = models[best_model_name]
    X_test_best = X_test_tree
    y_test_best = y_test_tree

# Predikcije
y_pred_best = best_model.predict(X_test_best)

# Confusion matrix
cm = confusion_matrix(y_test_best, y_pred_best)

# Vizualizacija confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Dissatisfied', 'Satisfied'],
            yticklabels=['Dissatisfied', 'Satisfied'])
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=12, fontweight='bold')
plt.ylabel('Actual', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n" + "="*80)
print(f"CLASSIFICATION REPORT - {best_model_name}")
print("="*80)
print(classification_report(y_test_best, y_pred_best))

# Feature importance za tree-based modele
if best_model_name in ['Decision Tree', 'Random Forest', 'XGBoost']:
    feature_importance = best_model.feature_importances_
    feature_names = X_test_best.columns
    
    # Kreiranje DataFrame-a
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    # Top 10 najvažnijih promenljivih
    top_10 = importance_df.head(10)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_10)), top_10['Importance'], color='skyblue')
    plt.yticks(range(len(top_10)), top_10['Feature'])
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.title(f'Top 10 najvaznijih promenljivih - {best_model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    print("\nTop 10 najvaznijih promenljivih:")
    print(importance_df.head(10).to_string(index=False))