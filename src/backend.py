import os
import certifi
from dotenv import load_dotenv

load_dotenv()


os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import TypedDict, Annotated
import operator
import uuid

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)


from langchain_nvidia_ai_endpoints import ChatNVIDIA
from src.tools.taivly_tool import tavily_search
from src.tools.flight_tool import search_flights    

def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set.")

    # if "sslmode=" not in database_url:
    #     seperator = "&" if "?" in database_url else "?"
    #     database_url += f"{seperator}sslmode=require"

    return database_url

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY environment variable is not set.")

#---- LLM -----

llm = ChatNVIDIA(
    model = "nvidia/nemotron-3.5-bolt-e4b",
)

#---- State -----

class TravelState(TypedDict):
    message : Annotated[list[AnyMessage], operator.add]
    user_query : str
    flight_results : str
    hotel_results : str
    itinerary : str
    llm_calls : int


# ---- flight agent -----

def flight_agent(state:TravelState):

    query = state["user_query"]
    flight_data = search_flights(query)

    return {
        "flight_results" : flight_data,
        "message" : [
            AIMessage(content= f"Flight result fetched")
        ],
        "llm_calls" : state.get("llm_calls", 0) + 1
    }


#---- Hotel agent -----

def hotel_agent(state:TravelState):

    query = state["user_query"]
    hotel_data = tavily_search(query)

    return {
        "hotel_results" : hotel_data,
        "message" : [
            AIMessage(content= f"Hotel result fetched")
        ],
        "llm_calls" : state.get("llm_calls", 0) + 1
    }

# ---- itinerary agent -----

def itinerary_agent(state:TravelState):

    prompt = f"""
    create a complete travel itinerary

    user query : {state['user_query']}
    flight results : {state['flight_results']}
    hotel results : {state['hotel_results']}


    Make the itinerary practical, budget-aware, and easy to follow.
    """

    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=prompt)
        ])

    return {
        "itinerary": response.content,
        "message" : [response],
        "llm_calls" : state.get("llm_calls", 0) + 1
    }

#---- final agent -----

def final_agent(state:TravelState):

    final_prompt = f"""
Generate the final travel response for the user.

User Request:
{state['user_query']}

Flights:
{state['flight_results']}

Hotels:
{state['hotel_results']}

Itinerary:
{state['itinerary']}

Format the final answer beautifully using these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Day-by-Day Itinerary
5. Estimated Budget
6. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight API may not provide ticket prices if pricing is unavailable.
- Keep the response useful for real travel planning.
"""

    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=final_prompt)
        ])

    return {
        "message" : [response],
        "llm_calls" : state.get("llm_calls", 0) + 1
    }


#---- Build Graph -----

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)

#----- Postgres checkpointer -----

DATABASE_URL = get_database_url()

_conn = psycopg.connect(
       DATABASE_URL,
        autocommit=True,
        sslmode="disable",
        row_factory=dict_row
)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)


#----- Function for Fast api ------

def run_travel_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        },
    }

    result = travel_graph.invoke(
    {
        "message" : [
            HumanMessage(content=user_input)
        ],
        "user_query" : user_input,
        "flight_results" : "",
        "hotel_results" : "",
        "itinerary" : "",
        "llm_calls" : 0
    },
        config=config
    )

    final_answer = result["message"][-1].content
    
    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0)
    }



