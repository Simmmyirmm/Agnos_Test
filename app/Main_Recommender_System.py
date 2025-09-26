import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from langdetect import detect
from googletrans import Translator
from thefuzz import fuzz
import json

def load_model_artefacts():
    print("Loading model artifacts.")
    similarity_model = pd.read_csv('../Model_Artefacts/symptom_similarity_model.csv', index_col=[0])
    print(similarity_model)
    with open("../Model_Artefacts/model_configs.json", "r", encoding="utf-8") as file:
        all_configs = json.load(file)

    model_artefacts = {
        "knowledge_base" : all_configs["symptom_knowledge_base"],
        "alias_to_official_map" : all_configs["alias_to_official"],
        "gender_rules" : all_configs["gender_rules"],
        "gender_boost_rules" : all_configs["gender_boost_rules"],
        "age_boost_rules" : all_configs["age_boost_rules"],
        'similarity_model': similarity_model
    }

    return model_artefacts

### Recommendations pipeline's helper functions
def extract_symptoms_from_search_terms(raw_text, knowledge_base, model_artefacts, confidence_threshold=70):
    if not isinstance(raw_text, str):
        return []
    
    ALIAS_TO_OFFICIAL_MAP = model_artefacts['alias_to_official_map']
    clean_text = raw_text.strip().lower()
    raw_text_split = raw_text.split(', ')
    for i in raw_text_split:
        if i in ALIAS_TO_OFFICIAL_MAP:
            return [ALIAS_TO_OFFICIAL_MAP[i]]

    found_symptoms = set()
    for official_symptom, aliases in knowledge_base.items():
        sorted_aliases = sorted(aliases, key=len, reverse=True)
        for alias in sorted_aliases:
            score = fuzz.partial_ratio(alias.lower(), clean_text)
            if score >= confidence_threshold:
                found_symptoms.add(official_symptom)
                break
    found_symptoms = sorted(found_symptoms, reverse=True)
    return list(found_symptoms)


def filter_and_rerank(recommendations, model_artefacts, user): 
    # We have two approaches, Pre-Filter (find a small group of patients who are very similar to our current user (e.g., females aged 30-40)) and Post-Filter. 
    # In this case, I decided to use Post-Filter because we have small-sized datasets/samples.
    GENDER_RULES = model_artefacts['gender_rules']
    GENDER_BOOST_RULES = model_artefacts['gender_boost_rules']
    AGE_BOOST_RULES = model_artefacts['age_boost_rules']
    final_recs = {}
    for symptom, score in recommendations.items():
        
        # Gender Filtering
        symptoms_to_remove = GENDER_RULES.get(user["gender"], [])
        if symptom in symptoms_to_remove:
            continue
        
        # Gender Boosting
        if symptom in GENDER_BOOST_RULES:
            rule = GENDER_BOOST_RULES[symptom]
            if user["gender"] == rule.get("gender"):
                score *= rule.get("boost_factor")
        
            
        # Age Boosting
        if symptom in AGE_BOOST_RULES:
            rule = AGE_BOOST_RULES[symptom]
            if user["age"] >= rule.get("min_age"):
                score *= rule.get("boost_factor")
        
        final_recs[symptom] = score
        
    sorted_recs = dict(sorted(final_recs.items(), key=lambda item: item[1], reverse=True))
    return sorted_recs

# Main Recommender Function
def get_recommendations_pipeline(search_term, model_artefacts, age=None, gender=None, top_n=5):
    similarity_df = model_artefacts['similarity_model']
    symptom_vocab_with_aliases = model_artefacts['knowledge_base']
    # Normalised user_search_term to Official Symptoms
    user_search_term_sypmtom = extract_symptoms_from_search_terms(search_term, symptom_vocab_with_aliases, model_artefacts)  
    
    # Recommend Next Official Symptoms to user
    final_scores = {}
    for symptom in user_search_term_sypmtom:
        if symptom in similarity_df.columns:
            similar_scores = similarity_df[symptom]
            for rec_symptom, score in similar_scores.items():
                if rec_symptom not in user_search_term_sypmtom:
                    final_scores[rec_symptom] = final_scores.get(rec_symptom, 0) + score
    
    for symptom in final_scores:
        final_scores[symptom] /= len(user_search_term_sypmtom)
        
    candidate_recs = dict(sorted(final_scores.items(), key=lambda item: item[1], reverse=True))

    # User's Personalised Next Official Symptoms
    if (age != None) and (gender != None):
        user_profile = {"age": age, "gender": gender}
        personalized_recs = filter_and_rerank(candidate_recs, model_artefacts, user_profile)
        return {
            'initial_user_search_term': search_term,
            'official_user_symptom_from_search_term': user_search_term_sypmtom,
            'next_symptom_recommendations': list(personalized_recs.keys())[:top_n]
        }
    
    return {
        'initial_user_search_term': search_term,
        'official_user_symptom_from_search_term': user_search_term_sypmtom,
        'next_symptom_recommendations': list(candidate_recs.keys())[:top_n]
    }