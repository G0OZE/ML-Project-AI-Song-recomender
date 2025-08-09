"""Simple Streamlit UI for demo recommendations."""
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.title("Demo Song Recommender")
user_id = st.text_input("User ID", "u1")
k = st.number_input("Number of tracks", value=5, step=1)

if st.button("Get Recommendations"):
    resp = requests.post(f"{API_URL}/recommend", json={"user_id": user_id, "k": int(k)})
    if resp.status_code == 200:
        for item in resp.json()["tracks"]:
            st.write(f"{item['title']} (score {item['score']:.2f})")
    else:
        st.error("API error")
