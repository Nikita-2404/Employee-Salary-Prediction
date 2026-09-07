📌 Project Overview

Employee Salary Prediction is a Machine Learning web application that predicts an employee's estimated annual salary based on their work-related information.

The application uses a Random Forest Regressor to predict salary from four numerical features: years of experience, performance score, number of certifications, and weekly working hours.

🎯 Objectives
Predict employee salary using Machine Learning.
Use employee-related numerical features for prediction.
Train a Random Forest regression model.
Provide a simple and interactive web interface.
Display the predicted salary in Indian Rupees (₹).
🛠️ Technologies Used
Python
Pandas
Scikit-learn
Flask
HTML
CSS
JavaScript
Tailwind CSS

The frontend is included directly inside app.py using Flask's render_template_string(), so separate templates and static folders are not required.

🤖 Machine Learning Model

The project uses a Random Forest Regressor with:

50 decision trees
Maximum depth of 12
Random state of 42

The model uses StandardScaler as part of a Scikit-learn pipeline.

Features Used
Feature	Description
years_experience	Years of employee experience
performance_score	Employee performance score
certifications_count	Number of certifications
work_hours_per_week	Weekly working hours
salary	Target salary

The application reads these columns from employee_salary_dataset.csv.

📂 Project Structure
Employee-Salary-Prediction/
│
├── app.py
├── employee_salary_dataset.csv
├── requirements.txt
└── README.md
🔄 Project Workflow
Employee Dataset
       ↓
Data Loading
       ↓
Feature Selection
       ↓
Standard Scaling
       ↓
Random Forest Regression
       ↓
Model Training
       ↓
User Input
       ↓
Salary Prediction
       ↓
Display Predicted Salary
🌐 Web Application

The Flask application provides an interactive salary prediction interface where users enter:

Years of Experience
Performance Score
Certifications Count
Weekly Work Hours

The prediction is then displayed as an estimated annual salary.

▶️ How to Run the Project
1. Clone the repository
git clone https://github.com/Nikita-2404/Employee-Salary-Prediction.git
2. Open the project
cd Employee-Salary-Prediction
3. Install required libraries
pip install -r requirements.txt
4. Run the application
python app.py
5. Open in browser

Go to:

http://127.0.0.1:5000/

The application runs Flask on port 5000.

✨ Features
🤖 Machine Learning salary prediction
🌐 Flask web application
📊 Random Forest regression
🎨 Interactive responsive interface
🌙 Dark/light theme
💰 Salary displayed in Indian Rupees
📁 CSV-based dataset

👩‍💻 Author
Nikita

📄 License
This project is created for educational and learning purposes.
