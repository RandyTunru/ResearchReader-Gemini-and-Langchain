# Research Reader with Gemini 2.5 and Langchain

A small, focused Streamlit app that answers questions only from the provided set of documents / research papers using a RAG pipeline powered by LangChain and Gemini 2.5 Pro.

## Quick start

1. Create & activate the environment (Conda):

    ```conda create --name researchreader --file environment.yml
    conda activate researchreader

2. Copy the example environment file and add your API key:

    ```cp env.example .env
    # then open .env and paste your Google Cloud / Gemini API key into the proper variable
    # (see env.example for the exact variable name used by this project)

3. Run the Streamlit app:

    `streamlit run app/app.py`

## Description

This project builds a Retrieval-Augmented Generation (RAG) pipeline using LangChain, using Gemini 2.5 Pro as the LLM to generate answers constrained to the supplied documents and providing a Streamlit UI for asking questions and inspecting sources / citations.

## Key Learning Points

- RAG pipeline fundamentals
- How to orchestrate retrieval + LLM responses with LangChain
- Integrating a (commercial) LLM — Gemini 2.5 Pro — for focused, source-backed answers

## Pre-requisites

- Python 3.10+ (recommended)
- Conda or another virtual environment manager
- A valid Google Cloud / Gemini API key (or whatever provider key is used in env.example)