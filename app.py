###################################################################################################
#  Brian Lesko               2023-11-14
#  This code implements an embedding shema that is used to compare the similarity of textual data.
#  Think of it as an upgraded Cmd+F search. Written in pure Python & created for learning purposes.
###################################################################################################

import pandas as pd
import numpy as np
import streamlit as st
import tiktoken as tk
from sklearn.metrics.pairwise import cosine_similarity
from customize_gui import gui
from api_key import openai_api_key
from openai import OpenAI
api_key = openai_api_key
client = OpenAI(api_key = api_key)
gui = gui()

def get_embedding(text, model="text-embedding-ada-002",encoding_format="float"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input = text, model=model)
    #st.write(response.data[0].embedding) # Debug : why the hell did OpenAI structure it like this? 
    return response.data[0].embedding

@st.cache_resource
def tokenize(text):
    enc = tk.encoding_for_model("gpt-3.5-turbo")
    tokens = enc.encode(text)
    return tokens

@st.cache_resource
def chunk_tokens(tokens, chunk_length=40, chunk_overlap=10):
    chunks = []
    for i in range(0, len(tokens), chunk_length - chunk_overlap):
        chunks.append(tokens[i:i + chunk_length])
    return chunks

@st.cache_resource
def detokenize(tokens):
    enc = tk.encoding_for_model("gpt-4")
    text = enc.decode(tokens)
    return text

@st.cache_resource
def embed_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        embeddings.append(get_embedding(chunk))
    return embeddings

def get_text(upload):
    # if the upload is a .txt file
    if upload.name.endswith(".txt"):
        document = upload.read().decode("utf-8")
    return document

def augment_query(contexts, query):
    augmented_query = (
        f"###Search Results: \n{contexts} #End of Search Results\n\n-----\n\n {query}" + """ In your answer please be clear and concise, sometime funny.
        If you need to make an assumption you must say so."""
    )
    return augmented_query

def generate_response(augmented_query,query):
    st.session_state.messages.append({"role": "user", "content": augmented_query})
    response = openai.ChatCompletion.create(model="gpt-4", messages=st.session_state.messages)
    # delete the last message from the session state, so that only the prompt and response are displayed on the next run: no context
    st.session_state.messages.pop()
    st.session_state.messages.append({"role": "user", "content": query})
    msg = response.choices[0].message
    st.session_state.messages.append(msg)
    return msg

class document:
    def __init__(self, name, text):
        self.name = name
        self.text = text
        self.tokens = tokenize(text)
        self.token_chunks = chunk_tokens(self.tokens, chunk_length=50, chunk_overlap=10)
        self.text_chunks = [detokenize(chunk) for chunk in self.token_chunks]
        self.chunk_embeddings = embed_chunks(self.text_chunks)
        self.embedding = get_embedding(self.text)
        self.df = pd.DataFrame({
            "name": [self.name], 
            "text": [self.text], 
            "embedding": [self.embedding], 
            "tokens": [self.tokens], 
            "token_chunks": [self.token_chunks], 
            "text_chunks": [self.text_chunks], 
            "chunk_embeddings": [self.chunk_embeddings]
            })

    def similarity_search(self, query, n=3):
        query_embedding = get_embedding(query)
        similarities = []
        for chunk_embedding in self.chunk_embeddings:
            similarities.append(cosine_similarity([query_embedding], [chunk_embedding])[0][0])
        # the indicies of the top n most similar chunks
        idx_sorted_scores = np.argsort(similarities)[::-1]
        context = ""
        for idx in idx_sorted_scores[:n]:
            context += self.text_chunks[idx] + "\n"
        return context
    
    def similarity(self, doc):
        return cosine_similarity([self.embedding], [doc.embedding])[0][0]

def main():
    gui.clean_format()
    with st.sidebar:
        gui.about(text="This code implements text embedding, check it out!")
    gui.display_existing_messages(intro = "Hi, I'm going to help you understand what an embedding is - and why it's useful. Let's get started by entering some text to embed.")
    text = st.chat_input("Write a message")
    if text:
        doc = document("User Input", text) # document class defined above
        with st.sidebar:
            st.markdown("""---""")
            st.subheader("Your text:")
            st.write(doc.text)
            st.write("Model used: text-embedding-ada-002")
        with st.chat_message("assistant"):
            st.write("Your text was embedded, here's the result: ")
            st.dataframe(np.array(doc.embedding).reshape(1, -1))
        with st.chat_message("assistant"):
            st.markdown(f"""
                Here's why it's useful
                - Searching through documents.
                - Comparing the similarity of two pieces of text.
                - Text generation - think ChatGPT.
                """) 
        with st.chat_message("assistant"):
            length = np.array(doc.embedding).shape
            st.markdown(f"""
                Here's how it works
                - Each time text is embedded, the result is a vector of values that 'describe' the text.
                - No matter the input text, the output is always the same length: {length}
                - For this reason, text is often chunked into smaller pieces, and each piece is embedded.
                - Similarity between two embeddings is calculated using L2 distance or cosine similarity. 
                """)
        with st.chat_message("assistant"):
            if "baseball_similarities" not in st.session_state:
                st.session_state.baseball_similarities = []
            if "football_similarities" not in st.session_state:
                st.session_state.football_similarities = []
            # Example of similarity
            baseball = document("Baseball", "Baseball is a sport where two teams of nine players take turns batting and fielding across nine innings. Each team's goal is to score more runs while batting than the other. Fielding plays a crucial role in preventing the opposing team from scoring runs. Baseball, often referred to as America's pastime, is played on a diamond-shaped field, with bases positioned at each corner.")
            football = document("Football", "The game is played in a series of plays or downs. There are four downs in football, and the team in possession of the ball (the offense) must advance the ball at least 10 yards within these four downs to maintain possession. If the offense fails to advance the ball the required distance, the ball is turned over to the opposing team (the defense).")
            baseball_similarity = doc.similarity(baseball)  
            football_similarity = doc.similarity(football)
            st.session_state.baseball_similarities.append(baseball_similarity)
            st.session_state.football_similarities.append(football_similarity)  
            st.markdown(f"""
                For example, even though all this text is hard coded, I can tell you if your text input is more simialar to baseball or football.
                """)
            # plot a 2d plot of the similarity between baseball on the X and the similarity to soccer on the Y
            st.write(f"Similarity to baseball: {baseball_similarity}")
            st.write(f"Similarity to football: {football_similarity}")

            import plotly as py
            fig = py.graph_objs.Figure()
            fig.add_trace(py.graph_objs.Scatter(x=st.session_state.baseball_similarities, y=st.session_state.football_similarities, mode='markers'))
            fig.update_layout(
                title="Similarity to Baseball vs. Similarity to Football",
                xaxis_title="Similarity to Baseball",
                yaxis_title="Similarity to Football",
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="#7f7f7f"
                )
            )
            st.plotly_chart(fig)
            

main()