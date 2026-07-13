import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(page_title="WC 2026 Analytics", page_icon="🏆", layout="wide")
st.title("🏆 World Cup 2026 Analytics Platform")


@st.cache_data
def load_match_data():
    url = 'https://raw.githubusercontent.com/martj42/international_results/master/results.csv'
    df = pd.read_csv(url)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'].dt.year >= 2000]

    def get_target(row):
        if row['home_score'] > row['away_score']: return 1
        elif row['home_score'] < row['away_score']: return 2
        else: return 0
    df['target'] = df.apply(get_target, axis=1)
    df['neutral'] = df['neutral'].astype(int)

    home_goals = df.groupby('home_team')['home_score'].mean().rename('home_attack')
    away_goals = df.groupby('away_team')['away_score'].mean().rename('away_attack')
    home_defense = df.groupby('home_team')['away_score'].mean().rename('home_defense')
    away_defense = df.groupby('away_team')['home_score'].mean().rename('away_defense')

    df = df.merge(home_goals, on='home_team', how='left')
    df = df.merge(away_goals, on='away_team', how='left')
    df = df.merge(home_defense, on='home_team', how='left')
    df = df.merge(away_defense, on='away_team', how='left')
    df = df.dropna()
    
    return df, home_goals, away_goals, home_defense, away_defense

@st.cache_data
def load_player_data():
    player_stats = pd.read_csv('player_stats.csv')
    squads = pd.read_csv('squads_and_players.csv')
    teams = pd.read_csv('teams.csv')
    
    # συνένωση των δεδομένων βάσει των ID
    df_players = pd.merge(squads, teams[['team_id', 'team_name']], on='team_id', how='left')
    df_players = pd.merge(
        df_players, 
        player_stats[['player_id', 'matches_played', 'minutes_played', 'goals', 'assists', 'yellow_cards', 'red_cards']], 
        on='player_id', 
        how='left', 
        suffixes=('_career', '_tournament')
    )
    
    # υπολογισμός ηλικίας (με βάση τον Ιούλιο του 2026)
    df_players['date_of_birth'] = pd.to_datetime(df_players['date_of_birth'])
    df_players['age'] = (pd.Timestamp('2026-07-01') - df_players['date_of_birth']).dt.days // 365
    
    # αναπλήρωση κενών τιμών
    df_players = df_players.fillna(0)
    
    return df_players

# φόρτωση δεδομένων
df_matches, home_goals, away_goals, home_defense, away_defense = load_match_data()
try:
    df_players = load_player_data()
except Exception as e:
    st.error("Σφάλμα: Δεν βρέθηκαν τα αρχεία των παικτών. Βεβαιώσου ότι τα 3 CSV βρίσκονται στον ίδιο φάκελο με το app.py!")
    st.stop()


tab1, tab2 = st.tabs(["⚽ Match Predictor", "🔎 Player Scouting"])


with tab1:
    st.header("Πρόβλεψη Αποτελέσματος (Μουντιάλ 2026)")
    with st.expander("Πώς λειτουργεί η πρόβλεψη;"):
        st.write("""
        Η πρόβλεψη βασίζεται σε ένα μοντέλο **Logistic Regression**, εκπαιδευμένο σε ιστορικά δεδομένα διεθνών αγώνων από το 2000 και μετά. 
        Το μοντέλο αξιολογεί τέσσερις βασικούς δείκτες δυναμικότητας για κάθε ομάδα: **Επίθεση Εντός**, **Άμυνα Εντός**, **Επίθεση Εκτός** και **Άμυνα Εκτός**. 
        
        Για να διασφαλίσουμε δίκαια αποτελέσματα σε ουδέτερα γήπεδα (όπως στο Μουντιάλ), εφαρμόζουμε τη μέθοδο **Test-Time Augmentation (TTA)**: το μοντέλο εκτελείται δύο φορές για κάθε ζευγάρι (με αντιστροφή της σειράς των ομάδων) και επιστρέφει τον **μέσο όρο των πιθανοτήτων**. Αυτό εξαλείφει κάθε στατιστική προκατάληψη που θα μπορούσε να προκύψει από τη θέση της ομάδας στο dropdown menu.
        """)
    st.write("Αλγόριθμος Logistic Regression προσαρμοσμένος για ουδέτερα γήπεδα.")

    features = ['home_attack', 'away_attack', 'home_defense', 'away_defense', 'neutral']
    X = df_matches[features]
    y = df_matches['target']
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    idx_0, idx_1, idx_2 = list(model.classes_).index(0), list(model.classes_).index(1), list(model.classes_).index(2)

    teams_df = pd.read_csv('teams.csv')
    teams_list = sorted(teams_df['team_name'].unique())
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Ομάδα 1", teams_list, index=teams_list.index('Argentina') if 'Argentina' in teams_list else 0)
    with col2:
        team2 = st.selectbox("Ομάδα 2", teams_list, index=teams_list.index('France') if 'France' in teams_list else 1)

    if st.button("Πρόβλεψη Αποτελέσματος", use_container_width=True):
        if team1 == team2:
            st.warning("Παρακαλώ επίλεξε δύο διαφορετικές ομάδες!")
        else:
            t1_h_att, t1_h_def = home_goals[team1], home_defense[team1]
            t1_a_att, t1_a_def = away_goals[team1], away_defense[team1]
            t2_h_att, t2_h_def = home_goals[team2], home_defense[team2]
            t2_a_att, t2_a_def = away_goals[team2], away_defense[team2]

            hosts_2026 = ['United States', 'Canada', 'Mexico']
            
            if team1 in hosts_2026:
                feat = pd.DataFrame([[t1_h_att, t2_a_att, t1_h_def, t2_a_def, 0]], columns=features)
                probs = model.predict_proba(feat)[0]
                prob_team1, prob_team2, prob_draw = probs[idx_1], probs[idx_2], probs[idx_0]
                st.caption(f"🏟️ *Πλεονέκτημα Έδρας για: {team1} (Host Nation).*")
            elif team2 in hosts_2026:
                feat = pd.DataFrame([[t2_h_att, t1_a_att, t2_h_def, t1_a_def, 0]], columns=features)
                probs = model.predict_proba(feat)[0]
                prob_team1, prob_team2, prob_draw = probs[idx_2], probs[idx_1], probs[idx_0]
                st.caption(f"🏟️ *Πλεονέκτημα Έδρας για: {team2} (Host Nation).*")
            else:
                feat_A = pd.DataFrame([[t1_h_att, t2_a_att, t1_h_def, t2_a_def, 1]], columns=features)
                feat_B = pd.DataFrame([[t2_h_att, t1_a_att, t2_h_def, t1_a_def, 1]], columns=features)
                probs_A, probs_B = model.predict_proba(feat_A)[0], model.predict_proba(feat_B)[0]
                prob_team1 = (probs_A[idx_1] + probs_B[idx_2]) / 2
                prob_team2 = (probs_A[idx_2] + probs_B[idx_1]) / 2
                prob_draw = (probs_A[idx_0] + probs_B[idx_0]) / 2
                st.caption("🏟️ *Ο αγώνας υπολογίστηκε σε Ουδέτερο Έδαφος (Test-Time Augmentation).*")

            total = prob_team1 + prob_team2 + prob_draw
            st.subheader("Στατιστικές Πιθανότητες:")
            c1, c2, c3 = st.columns(3)
            c1.success(f"🥇 Νίκη {team1}: {(prob_team1/total)*100:.1f}%")
            c2.info(f"🤝 Ισοπαλία: {(prob_draw/total)*100:.1f}%")
            c3.error(f"🥇 Νίκη {team2}: {(prob_team2/total)*100:.1f}%")



with tab2:
    st.header("Προφίλ Παίκτη & Αναζήτηση Σωσία (Player Similarity)")
    with st.expander("Πώς λειτουργεί η αναζήτηση σωσία;"):
        st.write("""
        Η μηχανή αναζήτησης χρησιμοποιεί τον αλγόριθμο **K-Nearest Neighbors (KNN)** με χρήση **Cosine Similarity** (συνημιτονική ομοιότητα). 
        
        Η διαδικασία περιλαμβάνει τρία βήματα:
        1. **Position Filtering:** Ο αλγόριθμος απομονώνει μόνο τους παίκτες που αγωνίζονται στην ίδια θέση με τον παίκτη-στόχο.
        2. **Standardization:** Τα στατιστικά στοιχεία (αξία αγοράς, συμμετοχές, γκολ κ.λπ.) κανονικοποιούνται (StandardScaler) ώστε κάθε χαρακτηριστικό να συνεισφέρει ισότιμα στον υπολογισμό, ανεξάρτητα από την κλίμακα των τιμών του.
        3. **Similarity Score:** Υπολογίζεται η μαθηματική απόσταση των διανυσμάτων στον πολυδιάστατο χώρο. 
        
        Το **Radar Chart** που εμφανίζεται οπτικοποιεί τη σύγκριση των προφίλ των δύο παικτών σε κλίμακα 0-100 (μέσω MinMaxScaler), προσφέροντας μια άμεση οπτική σύγκριση του στυλ παιχνιδιού τους.
        """)
    st.write("Ανάλυση των στατιστικών όλων των παικτών και εύρεση παρόμοιων προφίλ με K-Nearest Neighbors.")

    # επιλογή Παίκτη
    player_names = sorted(df_players['player_name'].unique())
    selected_player = st.selectbox("Επίλεξε Παίκτη", player_names)
    
    # ανάκτηση των δεδομένων του επιλεγμένου παίκτη
    player_data = df_players[df_players['player_name'] == selected_player].iloc[0]
    
    # προφιλ παιχτη
    st.subheader(f"👤 Προφίλ: {player_data['player_name']}")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Εθνική Ομάδα", player_data['team_name'])
    c2.metric("Σύλλογος", player_data['club_team'])
    c3.metric("Θέση", player_data['position'])
    c4.metric("Ηλικία / Ύψος", f"{int(player_data['age'])} ετών / {int(player_data['height_cm'])}cm")
    
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Αξία Αγοράς", f"€{player_data['market_value_eur']:,.0f}")
    c6.metric("Διεθνείς Συμμετοχές", int(player_data['caps']))
    c7.metric("Γκολ Καριέρας (Εθνική)", int(player_data['goals_career']))
    c8.metric("Λεπτά / Ασίστ Τουρνουά", f"{int(player_data['minutes_played'])}' / {int(player_data['assists'])}")

    # αναζητηση
    st.markdown("---")
    if st.button("Εύρεση Παρόμοιου Παίκτη", use_container_width=True):
        player_features = ['market_value_eur', 'caps', 'height_cm', 'goals_career', 'minutes_played', 'goals_tournament', 'assists']
        
        # φιλτράρισμα: συγκρίνουμε μόνο παίκτες στην ιδια θέση (π.χ. Μέσος με Μέσο)
        df_same_pos = df_players[df_players['position'] == player_data['position']].reset_index(drop=True)
        
        # εξαγωγή χαρακτηριστικών και κανονικοποίηση (Standardization)
        X_players_pos = df_same_pos[player_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_players_pos)
        
        # εκπαίδευση του KNN Model
        knn = NearestNeighbors(n_neighbors=2, metric='cosine')
        knn.fit(X_scaled)
        
        target_idx = df_same_pos[df_same_pos['player_name'] == selected_player].index[0]
        target_stats = X_scaled[target_idx].reshape(1, -1)
        
        distances, indices = knn.kneighbors(target_stats)
        
        if len(indices[0]) > 1:
            similar_idx = indices[0][1]
            similar_player = df_same_pos.iloc[similar_idx]
            sim_score = (1 - distances[0][1]) * 100

            st.success(f"🔍 Ο πιο παρόμοιος {player_data['position']} με τον **{selected_player}** είναι ο **{similar_player['player_name']}** ({similar_player['team_name']}) - Ομοιότητα: {sim_score:.1f}%")
            
            # δημιουργία Radar Chart με Plotly (μετατροπή κλίμακας στο 0-100 για οπτικοποίηση)
            min_max = MinMaxScaler(feature_range=(0, 100))
            X_radar = min_max.fit_transform(X_players_pos)
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=X_radar[target_idx], theta=player_features,
                fill='toself', name=selected_player, line_color='blue'
            ))
            fig.add_trace(go.Scatterpolar(
                r=X_radar[similar_idx], theta=player_features,
                fill='toself', name=similar_player['player_name'], line_color='red'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                title="Σύγκριση Στατιστικού Προφίλ"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Δεν βρέθηκαν αρκετοί παίκτες σε αυτή τη θέση για σύγκριση.")
