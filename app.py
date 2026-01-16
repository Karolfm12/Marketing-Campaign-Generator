import streamlit as st
import pandas as pd
from pycaret.clustering import *
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI
import plotly.express as px

# Ładowanie zmiennych środowiskowych
load_dotenv()

# Konfiguracja klienta OpenAI
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

# Konfiguracja strony
st.set_page_config(page_title="AI Marketing Campaign Designer", layout="wide")

# --- Funkcje AI (OpenAI API) ---

def get_ai_response(prompt, model="gpt-3.5-turbo"):
    """Wysyła zapytanie do OpenAI i zwraca treść."""
    if not client:
        return "⚠️ Brak klucza API OpenAI w pliku .env"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Jesteś kreatywnym dyrektorem marketingu z 20-letnim doświadczeniem. Mówisz po polsku."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Błąd API: {str(e)}"

def generate_cluster_name_ai(cluster_stats):
    """
    v4: Generuje kreatywną nazwę persony na podstawie statystyk.
    """
    # Zamiana statystyk na tekst czytelny dla modelu
    stats_text = cluster_stats.to_string()
    
    prompt = f"""
    Na podstawie poniższych średnich danych demograficznych/behawioralnych grupy klientów, stwórz krótką, chwytliwą nazwę dla tego segmentu (Persony).
    
    Dane:
    {stats_text}
    
    Zwróć TYLKO nazwę (max 3-4 słowa), bez cudzysłowów i zbędnych opisów.
    """
    return get_ai_response(prompt)

def generate_ad_content_ai(cluster_name, cluster_stats, goal):
    """
    v6: Generuje treść reklamową dopasowaną do persony i celu.
    """
    stats_text = cluster_stats.to_string()
    
    prompt = f"""
    Zaprojektuj treść posta reklamowego na social media.
    
    1. Grupa docelowa (Persona): "{cluster_name}"
    2. Charakterystyka grupy: {stats_text}
    3. Cel kampanii: "{goal}"
    
    Napisz krótki, angażujący tekst reklamowy (max 280 znaków) z emoji, który przekona tę konkretną grupę do realizacji celu.
    """
    return get_ai_response(prompt)

# --- Interfejs Użytkownika ---
st.title("Generator Kampani Marketingowych")

# Sprawdzenie klucza API
if not api_key:
    st.warning("⚠️ Nie wykryto klucza `OPENAI_API_KEY` w pliku `.env`. Aplikacja nie wygeneruje tekstów.")

# Sidebar - Panel sterowania
st.sidebar.header("1. Dane i Konfiguracja")

# v1 - Przesyłanie danych
uploaded_file = st.sidebar.file_uploader("Wgraj plik CSV z danymi klientów", type=["csv"])

if uploaded_file is None:
    st.info("👋 Wgraj plik CSV. Demo: Generuję losowe dane.")
    data = pd.DataFrame({
        'Wiek': np.random.randint(18, 70, 100),
        'Dochód_Roczny_k': np.random.randint(20, 150, 100),
        'Wynik_Wydatków_1-100': np.random.randint(1, 100, 100)
    })
else:
    data = pd.read_csv(uploaded_file)

with st.expander("📊 Podgląd danych wejściowych", expanded=False):
    st.dataframe(data.head())

# v2 - Liczba grup docelowych
num_clusters = st.sidebar.slider("Liczba grup docelowych", min_value=2, max_value=6, value=3)

# v5 - Cel kampanii
campaign_goal = st.sidebar.text_input("Cel kampanii reklamowej", "Promocja luksusowych wakacji zimowych")

# Przycisk uruchamiający proces
if st.sidebar.button("🚀 Projektuj Kampanię"):
    
    st.divider()
    
    # v3 - Trenowanie modelu (PyCaret)
    with st.spinner('1/3 Segmentuję klientów (Machine Learning)...'):
        s = setup(data, session_id=123, verbose=False, html=False)
        kmeans = create_model('kmeans', num_clusters=num_clusters)
        results = assign_model(kmeans)
    
    st.success("✅ Segmentacja zakończona!")
    
    # Sekcja Wyników
    st.subheader("🎯 Wyniki Kampanii")
    
    unique_clusters = sorted(results['Cluster'].unique())
    cols = st.columns(len(unique_clusters))
    
    # Przygotowanie paska postępu dla generowania AI
    progress_bar = st.progress(0)
    step = 1.0 / len(unique_clusters)
    
    for i, cluster_id in enumerate(unique_clusters):
        with cols[i % len(cols)]:
            cluster_data = results[results['Cluster'] == cluster_id]
            
            # Obliczanie średnich dla kontekstu AI
            stats = cluster_data.mean(numeric_only=True).drop('Cluster', errors='ignore').round(1)
            
            # v4 - Generowanie nazwy przez OpenAI
            with st.spinner(f'Analizuję grupę {cluster_id}...'):
                cluster_name = generate_cluster_name_ai(stats)
            
            st.markdown(f"### 🏷️ {cluster_name}")
            st.caption(f"ID Klastra: {cluster_id} | Liczebność: {len(cluster_data)}")
            
            # Wyświetlenie statystyk (jako mała tabelka lub json)
            st.markdown("**Profil:**")
            st.json(stats.to_dict())
            
            st.markdown("---")
            
            # v6 - Generowanie reklamy przez OpenAI
            with st.spinner('Piszę reklamę...'):
                ad_copy = generate_ad_content_ai(cluster_name, stats, campaign_goal)
            
            st.info(f"📢 **Reklama:**\n\n{ad_copy}")
        
        progress_bar.progress(min((i + 1) * step, 1.0))

    progress_bar.empty()
    
# --- NOWA SEKCJA: WIZUALIZACJA ---
    st.subheader("📊 Mapa Segmentów Klientów")
    
    # Tworzymy wykres punktowy (Scatter Plot)
    # Wybieramy dwie pierwsze kolumny do osi X i Y, a kolor uzależniamy od Klastra
    columns = data.select_dtypes(include=[np.number]).columns
    
    if len(columns) >= 2:
        fig = px.scatter(
            results, 
            x=columns[0], 
            y=columns[1], 
            color='Cluster',
            title=f"Podział klientów: {columns[0]} vs {columns[1]}",
            hover_data=columns, # Pokazuje wszystkie dane po najechaniu myszką
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Zbyt mało danych numerycznych, aby wygenerować wykres.")


    # Pobieranie wyników
    st.divider()
    csv = results.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Pobierz dane z segmentacją", csv, "segmentacja.csv", "text/csv")