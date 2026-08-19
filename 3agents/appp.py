import streamlit as st
import requests
import json
import concurrent.futures
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -----------------------------------------------------------------------------
# 1. Page Configuration & Title
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Intelligent Travel Planner",
    page_icon="✈️",
    layout="wide" 
)

st.title("✈️ Intelligent Travel & Cultural Planner")
st.caption("Powered by 3 Autonomous LangChain Agents, Groq LLM & Weather API")

# -----------------------------------------------------------------------------
# 2. API KEYS
# -----------------------------------------------------------------------------
# Groq API key is now entered by the user in the sidebar
st.sidebar.header("🔑 API Credentials")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
st.sidebar.markdown("---")

# Weather API key remains safely pulled from Streamlit Secrets
weather_api_key = st.secrets["WEATHER_API_KEY"]

# -----------------------------------------------------------------------------
# 3. Sidebar (Collapsible Left Menu)
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🧠 Agent Architecture")
st.sidebar.markdown("""
Behind the scenes, 3 specialized AI agents work together to plan your trip:

1. **Agent 1 (Weather Specialist)** 🌤️: 
   Checks real-time destination weather & provides a packing advisory.
2. **Agent 2 (Cultural Specialist)** 🏛️: 
   Evaluates cultural significance, local status & etiquette.
3. **Agent 3 (Master Concierge)** 🗺️: 
   Synthesizes data, calculates route logistics, generates the budget, and builds the final itinerary.
""")

# -----------------------------------------------------------------------------
# 4. Helper Functions for External APIs
# -----------------------------------------------------------------------------
def get_weather_data(city: str, api_key: str) -> dict:
    """Fetches real-time weather using OpenWeatherMap API."""
    if not api_key or api_key == "YOUR_WEATHER_API_KEY_HERE":
        return {
            "city": city,
            "temp": "22°C",
            "condition": "Partly Cloudy",
            "humidity": "60%",
            "note": "Standard seasonal weather estimation." 
        }
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=5)
        data = response.json()
        if response.status_code == 200:
            return {
                "city": city,
                "temp": f"{data['main']['temp']}°C",
                "condition": data['weather'][0]['description'].title(),
                "humidity": f"{data['main']['humidity']}%",
                "note": "Live data retrieved"
            }
        else:
            return {"error": data.get("message", "City not found")}
    except Exception as e:
        return {"error": str(e)}

# -----------------------------------------------------------------------------
# 5. Streamlit User Inputs (Custom Prompt)
# -----------------------------------------------------------------------------
user_prompt = st.text_area(
    "Enter your travel plan request:",
    placeholder="e.g., Plan a 1-day weekend road trip from Lahore to Murree, leaving Thursday night, including cost estimates...",
    height=120
)

plan_button = st.button("🚀 Generate Travel Plan", use_container_width=True)

# -----------------------------------------------------------------------------
# 6. Agent Workflow Execution
# -----------------------------------------------------------------------------
if plan_button:
    if not groq_api_key:
        st.error("⚠️ Please enter your Groq API Key in the sidebar to proceed.")
    elif not user_prompt.strip():
        st.warning("Please enter a prompt before proceeding.")
    else:
        try:
            # Fast model for initial routing and intermediate agents
            fast_llm = ChatGroq(
                model_name="openai/gpt-oss-20b",
                groq_api_key=groq_api_key,
                temperature=0.1
            )
            
            # Heavy model for final itinerary generation
            heavy_llm = ChatGroq(
                model_name="openai/gpt-oss-120b",
                groq_api_key=groq_api_key,
                temperature=0.2,
                max_tokens=4096  
            )

            # Guardrail & Intent Extraction Step (Using Fast LLM)
            guardrail_prompt = ChatPromptTemplate.from_template("""
                You are an intent classifier and entity extractor.
                User Prompt: "{user_prompt}"

                Task:
                1. Determine if the user's prompt is strictly related to travel, trip planning, vacation, or tourism.
                2. If it is NOT related to travel planning, output strictly: NOT_TRAVEL
                3. If it IS related to travel planning, extract the details in JSON format:
                {{
                    "is_travel": true,
                    "origin": "<origin city or 'Not Specified'>",
                    "destination": "<destination city/region or default to 'Murree'>",
                    "num_days": <extracted number of days as integer, default to 1 if unspecified>,
                    "travel_style": "<extracted style, e.g., 'Road Trip', 'Cultural Heritage', 'Adventure', 'Leisure', or 'General'>"
                }}

                Output ONLY the raw string "NOT_TRAVEL" or the pure JSON object. Do not include markdown code block backticks.
            """)

            guardrail_chain = guardrail_prompt | fast_llm | StrOutputParser()
            raw_validation = guardrail_chain.invoke({"user_prompt": user_prompt}).strip()

            if "NOT_TRAVEL" in raw_validation:
                st.warning("I am configured for travel planning requests only.")
            else:
                # Clean up any potential markdown formatting from JSON output
                clean_json_str = raw_validation.replace("```json", "").replace("```", "").strip()
                parsed_data = json.loads(clean_json_str)

                origin = parsed_data.get("origin", "Not Specified")
                destination = parsed_data.get("destination", "Murree")
                num_days = parsed_data.get("num_days", 1)
                travel_style = parsed_data.get("travel_style", "General")

                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("⚡ Agents 1 & 2 are analyzing climate and cultural data simultaneously...")
                progress_bar.progress(30)

                raw_weather = get_weather_data(destination, weather_api_key)

                # Prepare Weather Chain
                weather_prompt = ChatPromptTemplate.from_template("""
                    You are Agent 1: An expert Weather Forecaster and Packing Advisor.
                    Analyze the following weather data for {city}:
                    {weather_json}

                    Provide a concise summary containing:
                    1. General climate & temperature breakdown.
                    2. Recommended clothing, layers, and gear suitable for the destination's elevation and weather.
                    3. Best hours of the day for outdoor exploration and driving safety warnings (e.g., fog/rain).
                """)
                weather_chain = weather_prompt | fast_llm | StrOutputParser()

                # Prepare Cultural Chain
                cultural_prompt = ChatPromptTemplate.from_template("""
                    You are Agent 2: A Cultural Heritage and Local Experience Specialist.
                    Travel Route: Origin ({origin}) to Destination ({destination}).
                    Travel Style: {style}
                    User Request: {user_prompt}

                    Provide an insightful briefing:
                    1. **Heritage & Scenic Significance**: Key attractions and historical/natural relevance of {destination}.
                    2. **Local Norms & Safety**: Essential etiquette, road safety tips for mountain driving, and customs.
                    3. **Authentic Immersion**: Must-try local food, traditional stops along the route, and top spots to visit.
                """)
                cultural_chain = cultural_prompt | fast_llm | StrOutputParser()

                # RUN AGENT 1 AND AGENT 2 IN PARALLEL
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_weather = executor.submit(
                        weather_chain.invoke, 
                        {"city": destination, "weather_json": json.dumps(raw_weather)}
                    )
                    future_cultural = executor.submit(
                        cultural_chain.invoke, 
                        {"origin": origin, "destination": destination, "style": travel_style, "user_prompt": user_prompt}
                    )
                    
                # Extract results and strip reasoning tags so Streamlit renders them
                weather_analysis = future_weather.result().replace("<think>", "💡 **AI Reasoning:**\n> ").replace("</think>", "\n\n")
                cultural_analysis = future_cultural.result().replace("<think>", "💡 **AI Reasoning:**\n> ").replace("</think>", "\n\n")

                # Agent 3: Master Travel Concierge & Route Logistics
                status_text.text("⚡ Agent 3 (Master Travel Concierge) is generating final route & cost plan...")
                progress_bar.progress(70)

                master_prompt = ChatPromptTemplate.from_template("""
                    You are Agent 3: Lead Travel Concierge and Route Experience Specialist.
                    Synthesize the findings of Agent 1 and Agent 2 to satisfy the user's travel request:
                    User Request: "{user_prompt}"

                    Origin: {origin}
                    Destination: {destination}
                    Trip Duration: {num_days} Day(s)

                    --- AGENT 1 (WEATHER FINDINGS) ---
                    {weather_info}

                    --- AGENT 2 (CULTURAL & ROUTE FINDINGS) ---
                    {cultural_info}

                    --- STRICT RULES FOR ROUTING & FORMATTING ---
                    1. **NO EXACT DISTANCES**: NEVER state exact distances, kilometers (km), or miles anywhere in the response.
                    2. **Handling Origins**: 
                       - If the Origin is NOT "Not Specified", describe the travel between the origin and destination using estimated driving/transit duration in hours and recommended routes/highways.
                       - If the Origin IS "Not Specified", completely skip inter-city travel logistics and focus solely on the destination's daily itinerary, local transit, and experiences.
                    3. **Budget**: Provide practical cost estimates (meals, activities, local transit) without referencing exact kilometer calculations.
                    4. **NO HTML TAGS**: NEVER use `<br>` or any other HTML tags in your output. If you create tables, separate list items using commas or semicolons. If you need multi-line breakdowns, use standard Markdown bulleted lists outside of tables instead of forcing them into table cells.

                    --- REQUIRED OUTPUT FORMAT ---
                    Format your response in clean Markdown using the following exact headers:
                    ## 1. Trip & Route Overview
                    ## 2. Weather Advisory & Packing Guide
                    ## 3. Detailed Daily Itinerary
                    ## 4. Estimated Expense & Budget Breakdown
                    ## 5. Key Safety & Travel Tips
                """)
                master_chain = master_prompt | heavy_llm | StrOutputParser()
                
                # Setup Stream generator
                response_generator = master_chain.stream({
                    "user_prompt": user_prompt,
                    "origin": origin,
                    "destination": destination,
                    "num_days": num_days,
                    "weather_info": weather_analysis,
                    "cultural_info": cultural_analysis
                })

                progress_bar.progress(100)
                status_text.empty()

                # Display Results
                tab1, tab2, tab3 = st.tabs(["🗺️ Complete Itinerary & Logistics", "🌤️ Weather Specialist", "🏛️ Cultural & Route Report"])

                with tab1:
                    st.write_stream(response_generator)

                with tab2:
                    st.subheader("Agent 1: Weather Specialist Report")
                    st.json(raw_weather)
                    st.write(weather_analysis)

                with tab3:
                    st.subheader("Agent 2: Cultural & Route Report")
                    st.write(cultural_analysis)

        except Exception as e:
            st.error(f"An error occurred during execution: {str(e)}")