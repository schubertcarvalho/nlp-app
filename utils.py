"""
Created on Wed Nov 23 2022
@author: Schubert Carvalho

"""

# Core Pkgs
import streamlit as st


def write_title():
    """Write the main title to the app"""
    title_templ = """
    <div style="background-color:black;padding:8px;">
    <h1 style="color:red">BENlabs: Playing with NLP</h1>
    </div>
    """
    st.markdown(title_templ, unsafe_allow_html=True)

    # subheader_templ = """
    # <div style="background-color:back;padding:8px;">
    # <h3 style="color:white">Natural Language Processing On the Go...</h3>
    # </div>
    # """
    # st.markdown(subheader_templ, unsafe_allow_html=True)


def add_sidebar_logo():
    """Add logo to the sidebar"""
    st.sidebar.image("figs/tb-logo-lg.png", use_column_width=True)
