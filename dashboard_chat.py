import streamlit as st
from chat_engine import chat_with_cohere
from llm_backend import generate_response
import matplotlib.pyplot as plt
import pandas as pd

# Page setup
st.set_page_config(layout="wide")
st.title("🌊 ARGO Semantic Explorer")

# User input
user_input = st.text_input("Ask a question about ARGO floats:")

if user_input:
    # Chatbot response
    st.subheader("🧠 Chatbot Response")
    response_text, raw_data = chat_with_cohere(user_input)
    st.markdown(response_text)
    if "chart" in user_input.lower() and isinstance(raw_data, pd.DataFrame):

    # Show matched data
     if isinstance(raw_data, pd.DataFrame):
        st.subheader("📋 Matched Data")
        st.dataframe(raw_data)

        # Conditional chart rendering
        if any(word in user_input.lower() for word in ["chart", "plot", "graph", "visualize"]):
            if 'salinity' in raw_data.columns and 'temperature' in raw_data.columns:
                st.subheader("📊 Salinity vs Temperature")
                fig, ax = plt.subplots()
                ax.scatter(
                    raw_data['salinity'],
                    raw_data['temperature'],
                    c='blue',
                    edgecolors='k',
                    alpha=0.7
                )
                ax.set_xlabel("Salinity (PSU)")
                ax.set_ylabel("Temperature (°C)")
                ax.set_title("Semantic Matches: Salinity vs Temperature")
                st.pyplot(fig)
            else:
                st.warning("Salinity and temperature columns not found in the data.")
    else:
        st.info("No structured data returned for visualization.")


