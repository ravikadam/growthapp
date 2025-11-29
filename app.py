import streamlit as st
import json
import requests
import pandas as pd

# Configuration
OLLAMA_URL = "http://45.194.3.43:11434/api/generate"
MODEL_NAME = "gpt-oss:20b"
DATA_FILE = "output.jsonl"

def load_data():
    """Loads the flashpoint data from output.jsonl."""
    data = []
    try:
        with open(DATA_FILE, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    except FileNotFoundError:
        st.error(f"File {DATA_FILE} not found.")
        return []

def query_ollama(prompt, model=MODEL_NAME):
    """Sends a prompt to the Ollama API and returns the response."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with Ollama: {e}")
        return None

def get_flashpoint_prompt(history, data_context):
    return f"""
You are an expert analyst.
Here is a list of potential 'Flashpoints' with their IDs and titles:
{json.dumps(data_context, indent=2)}

Below is a conversation history between a User and an Assistant:
{history}

Task:
Identify the top 3 most likely Flashpoints that the User is facing based on the conversation.
For each shortlisted Flashpoint, provide:
1. The Flashpoint ID (srno).
2. The Title.
3. A Likelihood Score (1 to 5, where 5 is highest).
4. A brief explanation for the score.

Output Format (JSON):
[
  {{
    "srno": "FPx",
    "title": "...",
    "zone": "...",
    "score": 5,
    "explanation": "..."
  }},
  ...
]
Return ONLY the JSON array.
"""

def get_process_zone_prompt(history, data_context):
    # Extract unique zones for context
    zones = list(set(item['zone'] for item in data_context if item.get('zone')))
    
    return f"""
You are an expert analyst.
The available 'Process Zones' are:
{json.dumps(zones, indent=2)}

Below is a conversation history between a User and an Assistant:
{history}

Task:
Determine which Process Zone the User is most likely talking about or currently in.
Provide:
1. The Process Zone Name.
2. A Likelihood Score (1 to 5).
3. A brief explanation.

Output Format (JSON):
[
{{
  "zone": "...",
  "score": 5,
  "explanation": "..."
}}
]
Return ONLY the JSON object.
"""

def main():
    st.set_page_config(page_title="GrowthApp Chatbot", layout="wide")
    st.title("GrowthApp Chatbot")

    # Load Data
    if "data" not in st.session_state:
        st.session_state.data = load_data()

    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize Analysis Results
    if "flashpoints" not in st.session_state:
        st.session_state.flashpoints = []
    if "process_zone" not in st.session_state:
        st.session_state.process_zone = None

    # Layout: Chat (Left) vs Analysis (Right)
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Conversation")
        # specific height to make it scrollable - reduced to 400px as requested
        chat_container = st.container(height=400)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        if prompt := st.chat_input("Type your message here..."):
            # Add User Message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            # Analysis & Response
            with st.status("Analyzing conversation...", expanded=True) as status:
                
                # Trigger Analysis FIRST
                if st.session_state.data:
                    chat_history_str = "\\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                    
                    status.write("Identifying Flashpoints...")
                    # Flashpoint Analysis
                    fp_prompt = get_flashpoint_prompt(chat_history_str, st.session_state.data)
                    fp_response = query_ollama(fp_prompt)
                    try:
                        if fp_response:
                            start = fp_response.find('[')
                            end = fp_response.rfind(']') + 1
                            if start != -1 and end != -1:
                                st.session_state.flashpoints = json.loads(fp_response[start:end])
                    except Exception as e:
                        print(f"Error parsing flashpoints: {e}")

                    status.write("Determining Process Zone...")
                    # Process Zone Analysis
                    pz_prompt = get_process_zone_prompt(chat_history_str, st.session_state.data)
                    pz_response = query_ollama(pz_prompt)
                    try:
                        if pz_response:
                            start = pz_response.find('[')
                            end = pz_response.rfind(']') + 1
                            if start != -1 and end != -1:
                                st.session_state.process_zone = json.loads(pz_response[start:end])
                    except Exception as e:
                        print(f"Error parsing process zone: {e}")

                # Generate Assistant Response using Analysis Results
                with chat_container:
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        full_response = ""
                        
                        # Construct chat prompt with Identified Flashpoints context
                        if st.session_state.flashpoints:
                            # Use the identified flashpoints for the prompt
                            flashpoints_str = json.dumps(st.session_state.flashpoints, indent=2)
                            context_instruction = "Based on the analysis, the user is likely facing one of the following Flashpoints. Use this list to ask specific clarifying questions."
                        elif st.session_state.data:
                            # Fallback to full list if no specific flashpoints identified yet (or start of convo)
                            # But usually we want to narrow it down. Let's just provide the full list if empty?
                            # Or maybe just say "No specific flashpoints identified yet."
                            # User wants "shortlisted flashpoints and full flashpoints" - let's prioritize shortlisted.
                            flashpoint_titles = [item['title'] for item in st.session_state.data if 'title' in item]
                            flashpoints_str = json.dumps(flashpoint_titles, indent=2)
                            context_instruction = "Analyze the conversation against the full list of Flashpoints below."
                        else:
                            flashpoints_str = "No flashpoint data available."
                            context_instruction = ""

                        chat_history_str = "\\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                        
                        chat_prompt = f"""
You are an expert analyst and investigator.
Your goal is to identify which specific 'Flashpoint' (problem scenario) the user is facing.

{context_instruction}

Current Shortlisted/Potential Flashpoints:
{flashpoints_str}

Instructions:
1. Analyze the user's input and the conversation history.
2. If the user's situation is unclear or could match multiple flashpoints, ask a single, specific clarifying question to narrow it down.
3. Do NOT provide solutions, advice, or recommendations. Your ONLY job is to identify the problem.
4. Do NOT list the flashpoints to the user in chat message.
5. DO NOT tell user what are the identified Flashpoints
6. Keep your responses concise and professional.
7. Once Flashpoint identification is done, just say Thank you - we shall provide solution in upcoming version. DO NOT TELL WHICH FLASHPOINT IS IDENTIFIED

Conversation History:
{chat_history_str}

Assistant:
"""
                        status.write("Generating response...")
                        response_text = query_ollama(chat_prompt)
                        if response_text:
                            full_response = response_text
                            message_placeholder.markdown(full_response)
                            st.session_state.messages.append({"role": "assistant", "content": full_response})
                        else:
                            message_placeholder.markdown("Error generating response.")
                
                status.update(label="Analysis Complete", state="complete", expanded=False)
            
            # Rerun to update sidebar
            st.rerun()

    with col2:
        st.subheader("Real-time Analysis")
        
        st.markdown("### Likely Flashpoints")
        if st.session_state.flashpoints:
            for fp in st.session_state.flashpoints:
                with st.expander(f"{fp.get('srno')}: {fp.get('title')} ({fp.get('score')}/5)"):
                    st.markdown(f"**Zone:** {fp.get('zone', 'N/A')}")
                    st.write(fp.get('explanation'))
        else:
            st.write("Waiting for analysis...")

        st.divider()

        st.markdown("### Process Zone")
        if st.session_state.process_zone:
            # Ensure it's a list (handle backward compatibility if needed, though prompt changed)
            zones = st.session_state.process_zone if isinstance(st.session_state.process_zone, list) else [st.session_state.process_zone]
            for pz in zones:
                st.info(f"**{pz.get('zone', 'Unknown')}** (Score: {pz.get('score', 0)}/5)")
                st.caption(pz.get('explanation', ''))
        else:
            st.write("Waiting for analysis...")

if __name__ == "__main__":
    main()
