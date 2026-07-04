from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# ================================
# LOAD MODEL & COLUMNS
# ================================
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

# ================================
# PREPROCESS FUNCTION
# ================================
def preprocess(df):
    drop_cols = ['id', 'attack_cat', 'label']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)
    return df


# ================================
# ROUTE
# ================================
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    stats = None

    if request.method == 'POST':
        file = request.files.get('file')

        if file and file.filename.endswith('.csv'):
            try:
                df = pd.read_csv(file)
                total = len(df)

                processed = preprocess(df.copy())
                predictions = model.predict(processed)

                attack = int((predictions == 1).sum())
                normal = int((predictions == 0).sum())
                pct = round((attack / total) * 100, 2)

                stats = {
                    'total': total,
                    'attack': attack,
                    'normal': normal,
                    'pct': pct
                }

                if attack > 0:
                    result = f"ATTACK DETECTED — {attack} malicious packets ({pct}%)"
                else:
                    result = f"All Clear — {normal} packets are Normal"

            except Exception as e:
                result = f"Error: {str(e)}"

        else:
            result = "⚠️ Please upload a valid CSV file"

    return render_template('index.html', result=result, stats=stats)


# ================================
# RUN
# ================================
if __name__ == '__main__':
    app.run(debug=True)