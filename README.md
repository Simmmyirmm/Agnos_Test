Symptom Recommender System - Agnos Health Assignment
This repository contains the solution for the Data Science assignment from Agnos Health. It includes a complete pipeline for a next relevant symptom recommender system, from data processing and model training to a ready-to-use API.

Project Overview
The system is designed to predict the next likely symptom(s) for a patient based on their demographics and initial user search term. It follows a three-step pipeline:

Symptom Extraction (NLU): A fuzzy-matching model interprets the user's search_term input.

Candidate Generation: An Item-Item Cosine Similarity model finds general, official related symptoms.

Personalisation: A rule-based post-filtering system, filters and re-ranks recommendations based on the user's age and gender.

Directory Structure
agnos-recommender/
├── data/                  # Contains the test dataset
├── model_artifacts/       # Stores the pre-trained model and config files
├── app/                   # The production application code (FastAPI)
├── Main_Test_File.ipynb   # Jupyter Notebook for the main experiment for this assignment (Please see this file as the main one)
├── Main_Test_Agnos_Slide.pptx # Main Solution Overview and Explanation for this assignment
├── .gitignore             # Specifies files to be ignored by Git
├── README.md              # This documentation file
└── requirements.txt       # Python dependencies for the project

🚀 How to Set Up and Run
1. Prerequisites
Python 3.8+

pip and virtualenv

2. Setup & Installation
Clone the repository and set up a virtual environment.

# Clone the repository
git clone [<your-repo-url>](https://github.com/Simmmyirmm/Agnos_Test.git)
cd agnos-recommender

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required dependencies
pip install -r requirements.txt

3. Data and Training (Run Once)
Before running the API, you need to train the model.

Run the Training Notebook: Open and run all cells in the notebooks/1_data_prep_and_training.ipynb notebook. This will perform data cleaning, train the cosine similarity model, and save the final artifacts (symptom_similarity_model.csv and configs.json) into the model_artifacts/ directory.

4. Running the API
Once the model artifacts have been saved, you can run the API server.

# Navigate to the app directory
cd app

# Run the FastAPI server using Uvicorn
uvicorn Main_Test_API:app --reload

The API will be available at http://127.0.0.1:8000.

🧪 How to Test The Work
You can test the API in two ways:

1. Interactive API Docs (Recommended)
FastAPI automatically generates interactive documentation.

With the server running, open your web browser and navigate to http://127.0.0.1:8000/docs.

Expand the /recommend endpoint.

Click "Try it out" and modify the request body to test different scenarios.

Click "Execute" to see the live JSON response.

2. Using curl (Command Line)
You can also test the API from your terminal using a curl command.

Example Request (with personalisation):
curl -X POST "[http://127.0.0.1:8000/recommend](http://127.0.0.1:8000/recommend)" \
-H "Content-Type: application/json" \
-d '{
  "search_term": "ท้องอืด",
  "age": 47,
  "gender": "female"
}'

Example Request (without personalisation):
curl -X POST "[http://127.0.0.1:8000/recommend](http://127.0.0.1:8000/recommend)" \
-H "Content-Type: application/json" \
-d '{
  "search_term": "คันคอ, เจ็บคอ"
}'
