# Generative AI: Building RAG-Model using Orwells 1984

## Overview
The goal? To construct an intelligent system that combines the strengths of both retrieval-based and generative approaches. This guides you through building a Retrieval-Augmented Generation (RAG) model, using  George Orwell's "1984" as the knowledge base.

## Purpose
* Enhancing Text Understanding:  AI can deepen our understanding of literature. By building RAG models, the aim is to extract relevant information from a large corpus of text and generate insightful answers to user queries.
Seamless Integration of AI and Literature: AI companion that not only comprehends literary works but also engages in meaningful conversations about them, this project aims to bridge the gap between artificial intelligence and timeless literature.

## Key Steps
* Installation and Setup: We guide you through installing the necessary dependencies and setting up your environment.
* Data Preprocessing: We preprocess the text of “1984,” splitting it into chapters and preparing it for further analysis.
* Document Retrieval: Using a FAISS index, we retrieve relevant documents based on user queries.
* Answer Generation: Our RAG model generates answers by combining retrieved information with creative language generation.

## Why “1984”?
George Orwell’s dystopian masterpiece is rich in themes, symbolism, and thought-provoking ideas. And is in many ways more relevant today than ever. By building RAG models around it, we can unlock new ways to explore literature and engage with its content.

Let’s jump in!


## Key Findings

Focus on Conversation: Zephyr 7B Gemma is specifically fine-tuned for conversation tasks. This means it's better at understanding the nuances of natural language and generating coherent responses to questions. This is crucial for a RAG model that needs to interpret user queries and translate them into retrieving relevant information and formulating an answer.

# Steps Involved

## Flow Chart
![Flow Chart](./1984.jpg)

## Preprocessing
- Detailed explanation of the preprocessing function for the text data.
- Example output of the preprocessing step, showcasing the extracted chapters and parts from "1984".

## Model Training
- Instructions on how to train the RAG model using the preprocessed data.
- Tips on monitoring and improving model performance.

## Inference
- How to use the trained RAG model to answer questions about "1984".
- Examples of inference using the model.
