import os
import io
import re
import base64
import urllib.parse
from datetime import datetime
from collections import Counter

import feedparser
import pandas as pd
import pymongo
import psycopg2
import psycopg2.extras
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from dotenv import load_dotenv

# Load env file if exists
load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv(
    "MONGO_URI", 
    "mongodb+srv://arkan:mood123@cluster0.wrgqjch.mongodb.net/?appName=Cluster0"
)
# Supabase Postgres Configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres.eqhlekptvbcvznsmnorv:Ariftw242003@aws-1-ap-southeast-1.pooler.supabase.com:5432/postgres"
)

# Set matplotlib style
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.autolayout'] = True

def scrape_google_news():
    print("--- 1. SCRAPING GOOGLE NEWS ---")
    keywords = [
        "mood harian",
        "stress",
        "burnout",
        "overthinking",
        "kesehatan mental"
    ]
    
    semua_berita = []
    for keyword in keywords:
        safe_keyword = urllib.parse.quote(keyword)
        url = f"https://news.google.com/rss/search?q={safe_keyword}+when:1y&hl=id&gl=ID&ceid=ID:id"
        print(f"Scraping RSS untuk keyword: '{keyword}'")
        feed = feedparser.parse(url)
        
        for entry in feed.entries:
            semua_berita.append({
                "keyword": keyword,
                "title": entry.title,
                "link": entry.link,
                "published": entry.published,
                "source": entry.source.title if 'source' in entry else "Unknown"
            })
            
    print(f"Total berita ditemukan dari RSS: {len(semua_berita)}")
    return semua_berita

def save_news_to_mongodb(berita_list):
    print("--- 2. MENYIMPAN DATA KE MONGODB ATLAS ---")
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client["CapstoneBigData"]
        collection = db["Data_Mood_Harian"]
        
        # Simpan satu per satu agar tidak duplikat berdasarkan Link
        inserted_count = 0
        for item in berita_list:
            # check if exists
            exists = collection.find_one({"link": item["link"]})
            if not exists:
                collection.insert_one(item)
                inserted_count += 1
                
        print(f"Berhasil menyimpan {inserted_count} berita baru ke MongoDB.")
        
        # Ambil semua data berita mentah dari DB untuk data preparation
        data_mentah = list(collection.find({}, {'_id': 0}))
        print(f"Total berita di database sekarang: {len(data_mentah)}")
        return data_mentah
    except Exception as e:
        print("Gagal menghubungkan ke MongoDB untuk penyimpanan berita")
        print(e)
        return berita_list

def clean_text(teks):
    teks = str(teks).lower()
    teks = re.sub(r'[^a-z\s]', '', teks)
    return teks

def preprocess_news_data(data_mentah):
    print("--- 3. DATA PREPROCESSING (TEXT CLEANING & STOPWORD REMOVAL) ---")
    if not data_mentah:
        print("Data kosong. Menggunakan default data.")
        return pd.DataFrame()
        
    df = pd.DataFrame(data_mentah)
    df['clean_title'] = df['title'].apply(clean_text)
    
    factory = StopWordRemoverFactory()
    stopword_sastrawi = factory.create_stop_word_remover()
    
    stopword_tambahan = [
        'di', 'yang', 'dan', 'dari', 'ke', 'ini', 'itu', 'pada', 'dengan',
        'untuk', 'sebagai', 'hari', 'berita', 'juga', 'dalam', 'adalah',
        'oleh', 'atau', 'dapat', 'untuk', 'indonesia', 'news', 'google',
        'com', 'harian', 'saat', 'menurut', 'cara', 'bisa', 'ada', 'buat'
    ]
    
    def hapus_stopword(teks):
        teks = stopword_sastrawi.remove(teks)
        kata_kata = teks.split()
        kata_bersih = [
            kata for kata in kata_kata 
            if kata not in stopword_tambahan and len(kata) > 2
        ]
        return " ".join(kata_bersih)
        
    df['final_title'] = df['clean_title'].apply(hapus_stopword)
    print("Preprocessing selesai.")
    return df

def query_supabase_data():
    print("--- 4. QUERY DATA INTERNAL DARI SUPABASE POSTGRES ---")
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Query Mood Logs
        cursor.execute("SELECT mood_label, COUNT(*) as jumlah FROM mood_logs GROUP BY mood_label")
        mood_data = cursor.fetchall()
        
        # Query Emotion Detections
        cursor.execute("SELECT emotion_label, COUNT(*) as jumlah FROM emotion_detections GROUP BY emotion_label")
        emotion_data = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return mood_data, emotion_data
    except Exception as e:
        print("Gagal query data dari Supabase. Menggunakan fallback data.")
        print(e)
        return [], []

def generate_charts(df_news, mood_data, emotion_data):
    print("--- 5. GENERATING CHARTS & VISUALIZATIONS ---")
    
    charts_base64 = {}
    
    # helper function to convert plot to base64
    def get_base64_chart(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str

    # Color Palette
    primary_color = '#3A86F0' # MindCare Blue
    accent_color = '#FF006E'
    mood_colors = {
        'Sangat Baik': '#2EC4B6',
        'Baik': '#4EA8DE',
        'Biasa': '#FFB703',
        'Buruk': '#FB8500',
        'Sangat Buruk': '#E63946'
    }
    
    # -------------------------------------------------------------
    # 1. INTERNAL CHART 1: Mood Distribution
    # -------------------------------------------------------------
    print("- Membuat Grafik Mood Distribution...")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    labels = ['Sangat Buruk', 'Buruk', 'Biasa', 'Baik', 'Sangat Baik']
    counts = [0, 0, 0, 0, 0]
    
    # Map database data
    db_mood = {item['mood_label']: item['jumlah'] for item in mood_data}
    for i, label in enumerate(labels):
        counts[i] = db_mood.get(label, 0)
        
    # If no data, use beautiful mock data for display
    if sum(counts) < 5:
        counts = [3, 8, 25, 42, 18] # Fallback mock data
        title_suffix = " (Demo Data)"
    else:
        title_suffix = ""
        
    colors = [mood_colors[label] for label in labels]
    
    bars = ax.bar(labels, counts, color=colors, edgecolor='none', width=0.6)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2B2D42')
                    
    ax.set_title(f'Distribusi Mood Harian Pengguna{title_suffix}', fontsize=14, fontweight='bold', pad=15, color='#2B2D42')
    ax.set_ylabel('Jumlah Catatan', fontsize=12, color='#2B2D42')
    ax.set_xlabel('Kategori Mood', fontsize=12, color='#2B2D42')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#8D99AE')
    
    charts_base64['mood_distribution'] = get_base64_chart(fig)
    
    # -------------------------------------------------------------
    # 2. INTERNAL CHART 2: Emotion Detections Breakdown
    # -------------------------------------------------------------
    print("- Membuat Grafik Emotion Detection Breakdown...")
    fig, ax = plt.subplots(figsize=(6, 6))
    
    emotion_labels = []
    emotion_counts = []
    
    # Map database data
    for item in emotion_data:
        emotion_labels.append(item['emotion_label'])
        emotion_counts.append(item['jumlah'])
        
    # If no data or too few, use beautiful mock data
    if sum(emotion_counts) < 3:
        emotion_labels = ['Bahagia', 'Sedih', 'Netral', 'Marah', 'Cemas']
        emotion_counts = [45, 15, 20, 8, 12] # Fallback mock data
        title_suffix = " (Demo Data)"
    else:
        title_suffix = ""
        
    custom_colors = ['#2EC4B6', '#FFB703', '#9B5DE5', '#FB8500', '#F15BB5', '#00F5D4', '#00BBF9']
    
    ax.pie(
        emotion_counts, 
        labels=emotion_labels, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=custom_colors[:len(emotion_labels)],
        textprops={'fontsize': 11, 'fontweight': 'bold'},
        wedgeprops=dict(width=0.4, edgecolor='w') # Donut chart style
    )
    
    ax.set_title(f'Distribusi Deteksi Emosi Wajah{title_suffix}', fontsize=14, fontweight='bold', pad=15, color='#2B2D42')
    
    charts_base64['emotion_breakdown'] = get_base64_chart(fig)
    
    # -------------------------------------------------------------
    # 3. EXTERNAL CHART 1: Top 10 Most Common Words in News
    # -------------------------------------------------------------
    print("- Membuat Grafik Top 10 Words...")
    
    # default top words if data frame is empty
    top_words_df = pd.DataFrame(columns=['Kata', 'Frekuensi'])
    
    if not df_news.empty and 'final_title' in df_news.columns:
        semua_kata = " ".join(df_news['final_title']).split()
        hitung_kata = Counter(semua_kata)
        top_10_kata = hitung_kata.most_common(10)
        top_words_df = pd.DataFrame(top_10_kata, columns=['Kata', 'Frekuensi'])
        title_suffix = ""
    
    if top_words_df.empty or len(top_words_df) < 5:
        # Fallback mock data
        top_words_df = pd.DataFrame({
            'Kata': ['mental', 'kesehatan', 'stress', 'burnout', 'overthinking', 'anak', 'remaja', 'solusi', 'depresi', 'kelola'],
            'Frekuensi': [120, 115, 84, 62, 59, 45, 38, 30, 28, 25]
        })
        title_suffix = " (Demo Data)"
        
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Plot horizontal bar chart
    sns.barplot(
        x='Frekuensi',
        y='Kata',
        data=top_words_df,
        palette='Blues_r',
        ax=ax
    )
    
    # Add values to the bars
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        ax.text(width + 1.5, p.get_y() + p.get_height() / 2, f'{int(width)}', 
                ha='left', va='center', fontsize=10, fontweight='bold', color='#2B2D42')
                
    ax.set_title(f'Top 10 Kata Terbanyak pada Berita Kesehatan Mental{title_suffix}', fontsize=14, fontweight='bold', pad=15, color='#2B2D42')
    ax.set_xlabel('Frekuensi Kemunculan', fontsize=12, color='#2B2D42')
    ax.set_ylabel('Kata Kunci', fontsize=12, color='#2B2D42')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_color('#8D99AE')
    
    charts_base64['news_words'] = get_base64_chart(fig)
    
    # -------------------------------------------------------------
    # 4. EXTERNAL CHART 2: WordCloud
    # -------------------------------------------------------------
    print("- Membuat Grafik WordCloud...")
    
    if not df_news.empty and 'final_title' in df_news.columns:
        kata_gabungan = " ".join(" ".join(df_news['final_title']).split())
        title_suffix = ""
    else:
        kata_gabungan = "mental kesehatan stress burnout overthinking depresi emosi cemas psikologi konseling terapi remaja anak keluarga kelola bahagia tenang damai pulih"
        title_suffix = " (Demo Data)"
        
    wordcloud = WordCloud(
        width=900,
        height=450,
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate(kata_gabungan)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'WordCloud Topik Kesehatan Mental & Mood{title_suffix}', fontsize=15, fontweight='bold', pad=15, color='#2B2D42')
    
    charts_base64['news_wordcloud'] = get_base64_chart(fig)
    
    print("Visualisasi selesai dibuat.")
    return charts_base64

def save_charts_to_mongodb(charts_base64, news_count):
    print("--- 6. MENYIMPAN GRAFIK KE MONGODB ATLAS ---")
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client["CapstoneBigData"]
        collection = db["BigData_Charts"]
        
        chart_document = {
            "_id": "latest_charts",
            "updated_at": datetime.now(),
            "news_count": news_count,
            "chart_mood_distribution": charts_base64['mood_distribution'],
            "chart_emotion_breakdown": charts_base64['emotion_breakdown'],
            "chart_news_words": charts_base64['news_words'],
            "chart_news_wordcloud": charts_base64['news_wordcloud']
        }
        
        # Replace or insert
        collection.replace_one(
            {"_id": "latest_charts"}, 
            chart_document, 
            upsert=True
        )
        print("Berhasil mengupload chart terbaru ke MongoDB Atlas!")
    except Exception as e:
        print("Gagal mengupload chart ke MongoDB")
        print(e)

def save_charts_locally(charts_base64):
    print("--- 7. MENYIMPAN GRAFIK SECARA LOKAL ---")
    charts_dir = os.path.join(os.path.dirname(__file__), "charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    for name, b64_str in charts_base64.items():
        file_path = os.path.join(charts_dir, f"{name}.png")
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(b64_str))
        print(f"Tersimpan lokal: {file_path}")

def main():
    print("==================================================")
    print("MENGJALANKAN PIPELINE BIG DATA ANALYTICS MINDCARE")
    print("Waktu:", datetime.now().isoformat())
    print("==================================================")
    
    # 1. Scrape News
    berita_rss = scrape_google_news()
    
    # 2. Save news to MongoDB and get complete records
    data_mentah = save_news_to_mongodb(berita_rss)
    
    # 3. Preprocess news
    df_news = preprocess_news_data(data_mentah)
    
    # 4. Fetch internal data from Supabase
    mood_data, emotion_data = query_supabase_data()
    
    # 5. Generate Charts
    charts_base64 = generate_charts(df_news, mood_data, emotion_data)
    
    # 6. Save Charts to MongoDB Atlas
    save_charts_to_mongodb(charts_base64, len(data_mentah))
    
    # 7. Save Charts Locally
    save_charts_locally(charts_base64)
    
    print("==================================================")
    print("PIPELINE BERHASIL DIJALANKAN SEPENUHNYA! [OK]")
    print("==================================================")

if __name__ == "__main__":
    main()
