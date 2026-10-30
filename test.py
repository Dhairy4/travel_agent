from src.tools.taivly_tool import tavily_search
from src.tools.flight_tool import search_flights
from src.backend import run_travel_agent


# user_input = input("Enter your travel query: ")

response = run_travel_agent(
    user_input = "plan a 7 days trip in gujarat ahmedabad",
    thread_id = "test_thread_1"
                            )

print(response["answer"])


# res = search_flights("plan 7 days trip in gujarat ahmedabad")

# print(res)

# res = tavily_search("Best places to visit in Europe")

# print(res) 