import os
import openai
import json

from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

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

    for combination in winning_combinations:
        if all(p in history[player] for p in combination):
            return f"Player {player} wins"

    return "Nobody wins"


def get_completion(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=[
            {
                "name": "play",
                "description": "Call this function to indicate when a player played",
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
        temperature=0.1,
    )

    return response


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
or 5. If there are no available positions, the game is over and you should
write "The game is a draw."

Step 3. Remove the position you played from the board. For example, if the
board was [1, 2, 6, 7, 9] and you played in position 6, the new board will be
[1, 2, 7, 9]. If the board was [3, 4, 5] and you played in position 5, the new
board will be [3, 4].

Step 4. Call the function play to add the position you played to the history.
The function will return the name of the player who wins the game, or "Nobody
wins" if the game is not over yet.

Step 5. You are now the second player, named Alice. Repeat steps 2, 3, 4, and
5. Continue the game until one of the two players wins the game.
"""


messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "assistant", "content": "Board: [1, 2, 3, 4, 5, 6, 7, 8, 9]"},
    {"role": "user", "content": "You play first"},
]

while True:
    response = get_completion(messages)

    if response.choices[0]["finish_reason"] == "stop":
        print(response.choices[0].message["content"])
        break

    elif response.choices[0]["finish_reason"] == "function_call":
        fn_name = response.choices[0].message["function_call"].name
        args = response.choices[0].message["function_call"].arguments
        arguments = json.loads(args)

        result = locals()[fn_name](**arguments)
        print_board()

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
