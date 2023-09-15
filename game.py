import os
import openai
import json

from dotenv import load_dotenv
from comet_llm import Span, end_chain, start_chain

load_dotenv()

# You need to set your OpenAI and Comet API keys in a .env file.
openai.api_key = os.getenv("OPENAI_API_KEY")
COMET_API_KEY = os.getenv("COMET_API_KEY")


MODEL = "gpt-3.5-turbo-0613"

history = {}


def print_board():
    players = list(history.keys())

    board = ""
    for i in range(9):
        if i + 1 in history[players[0]]:
            board += "X"
        elif len(players) > 1 and i + 1 in history[players[1]]:
            board += "O"
        else:
            board += " "
        if (i + 1) % 3 == 0:
            board += "\n"
        else:
            board += "|"

    print(board)


def play(player, position):
    print(f"{player} played in position {position}")

    if player not in history:
        history[player] = []

    history[player].append(position)

    winning_combinations = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [1, 4, 7],
        [2, 5, 8],
        [3, 6, 9],
        [1, 5, 9],
        [3, 5, 7],
    ]

    player1_plays = len(history.get("Bob", []))
    player2_plays = len(history.get("Alice", []))

    if player1_plays + player2_plays == 9:
        return "The game is a draw."

    for combination in winning_combinations:
        if all(p in history[player] for p in combination):
            return f"Player {player} wins"

    return "Nobody wins"


def get_completion(messages, parameters):
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        functions=[
            {
                "name": "play",
                "description": "Call this function when a player plays",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "player": {
                            "type": "string",
                            "description": "The name of the player who played",
                        },
                        "position": {
                            "type": "integer",
                            "description": "The position where the player played",
                        },
                    },
                    "required": ["player", "position"],
                },
            },
        ],
        **parameters,
    )

    return response


def call_function(response, messages):
    fn_name = response.choices[0].message["function_call"].name
    args = response.choices[0].message["function_call"].arguments
    arguments = json.loads(args)

    with Span(
        category="llm-function-call",
        name=fn_name,
        inputs=arguments,
    ) as span:
        result = globals()[fn_name](**arguments)
        print_board()

        span.set_outputs({"output": result})

        messages.append(
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": fn_name,
                    "arguments": args,
                },
            }
        )

        messages.append(
            {
                "role": "function",
                "name": fn_name,
                "content": result,
            }
        )


SYSTEM_PROMPT = """
You will play a board game simulating two different players.

Here is how the game works:

The board has 9 positions represented by a number from 1 to 9,
eg. [1, 2, 3, 4, 5, 6, 7, 8, 9]. 

Repeat these steps until the game is over:

Step 1. You are the first player, named Bob.

Step 2. Choose any of the available positions from the board. You can only
pick one of the values in the current board. For example, if the board is
[1, 2, 6, 7, 9], you can play any of the following positions: 1, 2, 6, 7 or 9.
If the board is [3, 4, 5], you can play any of the following positions: 3, 4
or 5.

Step 3. Remove the position you played from the board. For example, if the
board was [1, 2, 6, 7, 9] and you played in position 6, the new board will be
[1, 2, 7, 9]. If the board was [3, 4, 5] and you played in position 5, the new
board will be [3, 4].

Step 4. Call the function play to add the position you played to the history.
The function will return the name of the player who wins the game, whether the
game is a draw or "Nobody wins" if the game is not over yet.

Step 5. You are now the second player, named Alice. Repeat steps 2, 3, 4, and
5. Continue the game until one of the two players wins the game or the game is
a draw.
"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "assistant", "content": "Board: [1, 2, 3, 4, 5, 6, 7, 8, 9]"},
    {"role": "user", "content": "You play first"},
]

parameters = {
    "temperature": 0.1,
}


start_chain(
    inputs={
        "prompt": SYSTEM_PROMPT,
    },
    api_key=COMET_API_KEY,
)


while True:
    with Span(
        category="llm-call",
        name="llm-generation",
        metadata={"model": MODEL, "parameters": parameters},
        inputs=messages,
    ) as span:
        response = get_completion(messages, parameters)

        if response.choices[0]["finish_reason"] == "stop":
            print(response.choices[0].message["content"])
            end_chain({"output": response.choices[0].message["content"]})
            break

        elif response.choices[0]["finish_reason"] == "function_call":
            call_function(response, messages)