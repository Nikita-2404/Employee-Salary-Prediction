import os
import pandas as pd
from flask import Flask, render_template_string, request, jsonify
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Global variable for model pipeline
model_pipeline = None

# Numeric feature column definitions
NUMERIC_FEATURES = ['years_experience', 'performance_score', 'certifications_count', 'work_hours_per_week']
TARGET_COL = 'salary'

def train_model_from_csv():
    """Reads dataset from CSV and trains model using ONLY numeric columns."""
    global model_pipeline
    csv_filename = 'employee_salary_dataset.csv'

    if not os.path.exists(csv_filename):
        raise FileNotFoundError(f"'{csv_filename}' not found in the project directory.")

    print(f"Loading dataset from {csv_filename}...")
    df = pd.read_csv(csv_filename)

    # Filter and use ONLY numeric columns
    X = df[NUMERIC_FEATURES]
    y = df[TARGET_COL]

    # Build numeric-only processing pipeline
    model_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1))
    ])

    print("Training Random Forest Regressor on numeric features...")
    model_pipeline.fit(X, y)
    print("Model training complete!")

# UI with Dynamic Animated Background & Glassmorphism UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Salary Predictor Pro</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        brand: { 500: '#6366f1', 600: '#4f46e5' }
                    }
                }
            }
        }
    </script>
    <style>
        body {
            background-size: 400% 400%;
            animation: gradientBg 15s ease infinite;
        }

        .light-bg {
            background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
            background-size: 400% 400%;
        }

        .dark-bg {
            background: linear-gradient(-45deg, #0f172a, #1e1b4b, #311042, #090d16);
            background-size: 400% 400%;
        }

        @keyframes gradientBg {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(80px);
            opacity: 0.5;
            animation: float 10s infinite alternate ease-in-out;
            pointer-events: none;
        }

        @keyframes float {
            0% { transform: translate(0, 0) scale(1); }
            100% { transform: translate(60px, 80px) scale(1.2); }
        }

        .glass {
            background: rgba(255, 255, 255, 0.45);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.4);
        }
        .dark .glass {
            background: rgba(15, 23, 42, 0.65);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .btn-active:active {
            transform: scale(0.96);
        }
    </style>
</head>
<body id="bgBody" class="dark-bg text-gray-800 dark:text-gray-100 min-h-screen transition-colors duration-500 flex flex-col justify-center items-center p-4 relative overflow-x-hidden">

    <!-- Glowing Background Orbs -->
    <div class="orb w-72 h-72 bg-indigo-500 top-10 left-10"></div>
    <div class="orb w-96 h-96 bg-purple-500 bottom-10 right-10" style="animation-delay: -5s;"></div>

    <!-- Header & Theme Toggle -->
    <div class="w-full max-w-2xl flex justify-between items-center mb-6 z-10">
        <div>
            <h1 class="text-3xl font-extrabold tracking-tight bg-gradient-to-r from-indigo-400 via-purple-300 to-pink-400 bg-clip-text text-transparent">
                Salary AI Engine
            </h1>
            <p class="text-sm text-gray-300 dark:text-gray-400">Numeric ML Compensation Estimator</p>
        </div>
        <button id="themeToggle" class="p-3 rounded-full glass hover:bg-white/20 dark:hover:bg-black/30 transition-all btn-active">
            <span id="themeIcon" class="text-xl">☀️</span>
        </button>
    </div>

    <!-- Main Card -->
    <div class="w-full max-w-2xl glass rounded-3xl p-8 shadow-2xl transition-all z-10">
        <form id="predictionForm" class="space-y-6">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                <!-- Experience -->
                <div>
                    <label class="block text-sm font-medium mb-2">Years of Experience</label>
                    <input type="number" step="0.1" name="years_experience" min="0" max="40" value="3.5" required
                        class="w-full px-4 py-3 rounded-xl bg-white/50 dark:bg-gray-900/60 border border-gray-300 dark:border-gray-700 focus:ring-2 focus:ring-indigo-500 focus:outline-none transition">
                </div>

                <!-- Performance Score -->
                <div>
                    <label class="block text-sm font-medium mb-2">Performance Score (1.0 - 5.0)</label>
                    <input type="number" step="0.1" name="performance_score" min="1.0" max="5.0" value="4.2" required
                        class="w-full px-4 py-3 rounded-xl bg-white/50 dark:bg-gray-900/60 border border-gray-300 dark:border-gray-700 focus:ring-2 focus:ring-indigo-500 focus:outline-none transition">
                </div>

                <!-- Certifications Count -->
                <div>
                    <label class="block text-sm font-medium mb-2">Certifications Count</label>
                    <input type="number" name="certifications_count" min="0" max="10" value="2" required
                        class="w-full px-4 py-3 rounded-xl bg-white/50 dark:bg-gray-900/60 border border-gray-300 dark:border-gray-700 focus:ring-2 focus:ring-indigo-500 focus:outline-none transition">
                </div>

                <!-- Weekly Work Hours -->
                <div>
                    <label class="block text-sm font-medium mb-2">Weekly Work Hours</label>
                    <input type="number" step="0.1" name="work_hours_per_week" min="10" max="80" value="40" required
                        class="w-full px-4 py-3 rounded-xl bg-white/50 dark:bg-gray-900/60 border border-gray-300 dark:border-gray-700 focus:ring-2 focus:ring-indigo-500 focus:outline-none transition">
                </div>

            </div>

            <!-- Submit Button -->
            <button type="submit" 
                class="w-full py-4 bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 hover:from-indigo-500 hover:to-pink-500 text-white font-bold rounded-xl shadow-lg hover:shadow-indigo-500/30 transition-all transform btn-active">
                Predict Salary
            </button>
        </form>

        <!-- Result Card -->
        <div id="resultContainer" class="hidden mt-8 p-6 rounded-2xl bg-indigo-500/10 border border-indigo-400/30 text-center transition-all backdrop-blur-md">
            <span class="text-sm font-semibold uppercase text-indigo-300 tracking-wider">Estimated Annual Salary</span>
            <div id="predictedSalary" class="text-4xl font-extrabold text-indigo-400 mt-2">₹0</div>
        </div>
    </div>

    <script>
        const themeToggleBtn = document.getElementById('themeToggle');
        const themeIcon = document.getElementById('themeIcon');
        const bgBody = document.getElementById('bgBody');

        themeToggleBtn.addEventListener('click', () => {
            if (document.documentElement.classList.contains('dark')) {
                document.documentElement.classList.remove('dark');
                bgBody.classList.remove('dark-bg');
                bgBody.classList.add('light-bg');
                themeIcon.textContent = '🌙';
            } else {
                document.documentElement.classList.add('dark');
                bgBody.classList.remove('light-bg');
                bgBody.classList.add('dark-bg');
                themeIcon.textContent = '☀️';
            }
        });

        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const payload = Object.fromEntries(formData.entries());

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const result = await response.json();
            const resultBox = document.getElementById('resultContainer');
            const salaryText = document.getElementById('predictedSalary');

            if (result.status === 'success') {
                salaryText.textContent = '₹' + Number(result.prediction).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
                resultBox.classList.remove('hidden');
            } else {
                alert('Error making prediction.');
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([{
            'years_experience': float(data['years_experience']),
            'performance_score': float(data['performance_score']),
            'certifications_count': int(data['certifications_count']),
            'work_hours_per_week': float(data['work_hours_per_week'])
        }])
        
        prediction = model_pipeline.predict(input_df)[0]
        return jsonify({'status': 'success', 'prediction': float(prediction)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    train_model_from_csv()
    app.run(host='0.0.0.0', port=5000, debug=True)