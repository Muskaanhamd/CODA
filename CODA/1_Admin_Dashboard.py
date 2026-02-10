import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="CODA Admin", page_icon="🔑", layout="wide")

st.title("🔑 CODA: Global Admin Dashboard")
st.markdown("Monitor system integrity and source reliability in real-time.")

if os.path.exists("coda_feedback_log.csv"):
    df = pd.read_csv("coda_feedback_log.csv")
    
    # --- TOP LEVEL METRICS ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Verification Requests", len(df))
    
    correct_count = len(df[df['user_feedback'] == 'Correct'])
    accuracy = (correct_count / len(df)) * 100
    col2.metric("System Confidence (User-Validated)", f"{accuracy:.1f}%")
    
    # --- SOURCE ANALYSIS ---
    st.markdown("---")
    st.subheader("🌐 News Source Reliability Matrix")
    
    # Extract all sources from the comma-separated strings
    all_sources = df['sources'].dropna().str.split(', ').explode()
    if not all_sources.empty:
        source_counts = all_sources.value_counts()
        
        col_chart, col_table = st.columns([2, 1])
        
        with col_chart:
            st.write("Most Frequent Truth-Providers")
            st.bar_chart(source_counts)
        
        with col_table:
            st.write("Source Frequency")
            st.dataframe(source_counts, use_container_width=True)
    
    # --- RAW LOGS ---
    st.markdown("---")
    st.subheader("📋 Recent Intelligence Logs")
    st.dataframe(df.tail(20), use_container_width=True)
    
    # Clear logs button (Security feature)
    if st.button("🗑️ Archive and Clear Logs"):
        os.rename("coda_feedback_log.csv", f"archive_{int(time.time())}.csv")
        st.rerun()

else:
    st.info("The Intelligence Matrix is currently empty. Feedback data will appear here once users begin verifying claims.")