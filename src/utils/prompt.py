import json

from src.datasets.dataset import Session

COT_PROMPT = f"""# Role

You are a helpful assistant that answers the user's question.

# Task

I will give you several history chats between you and a user. Please answer the question based on the relevant chat history. Answer the question step by step: first extract all the relevant information, and then reason over the information to get the answer.

# Chats History
%s

# Question

%s

# Answer (step by step)

"""

PROMPT = f"""# Role

You are a helpful assistant that answers the user's question.

# Task

I will give you several history chats between you and a user. Please answer the question based on the relevant chat history.

# Chats History
%s

# Question

%s

# Answer

"""


def get_prompt(question: str, chunks: list[str]) -> list[dict[str, str]]:
    if chunks:
        previous_conversations = "\n".join(
            [f'\n### Chat {i + 1}\n\n"""\n{conversation}\n"""' for i, conversation in enumerate(chunks)])
    else:
        previous_conversations = "No relevant chats history."
    prompt = PROMPT % (previous_conversations, question)
    return [{"role": "user", "content": prompt}]


def get_date_prompt(question: str, question_date: str) -> list[dict[str, str]]:
    system_prompt = "You will be given a question from a human user asking about some previous events, as well as the time the question is asked. Infer a potential time range such that the events happening in this range is likely to help to answer the question (a start date and an end date). Write a json dict two fields: \"start\" and \"end\". Write date in the form YYYY/MM/DD. If the question does not have any temporal references, do not attempt to guess a time range. Instead, just say N/A."
    user_prompt = "Question date: {}\nQuestion:\n{}\n\nRelevant Date Range(dict in json format; do not generate anything else):"
    examples = [
        ({'date': '2023/07/01 (Sat) 23:13',
          'question': 'What was the date on which I attended the first BBQ event in June?'},
         json.dumps({'start': "2023/06/01", "end": "2023/06/30"})),
        ({'date': '2023/04/10 (Mon) 08:05', 'question': 'Where did I attend the religious activity last week?'},
         json.dumps({"start": "2023/04/03", "end": "2023/04/09"})),
        ({'date': '2023/04/01 (Sat) 20:22',
          'question': 'What did I do with Rachel on the Wednesday two months ago?'},
         json.dumps({"start": "2023/01/25", "end": "2023/02/05"})),
        ({'date': '2023/05/30 (Tue) 01:50', 'question': 'Which pair of shoes did I clean last month?'},
         json.dumps({'start': "2023/04/01", "end": "2023/04/30"})),
        ({'date': '2023/04/18 (Tue) 02:06', 'question': 'Who did I meet with during the lunch last Tuesday?'},
         json.dumps({"start": "2023/04/10", "end": "2023/04/12"})),
        ({'date': '2023/05/27 (Sat) 01:55',
          'question': 'How many months ago did I book the Airbnb in San Francisco?'}, 'N/A'),
        ({'date': '2023/09/04 (Mon) 17:07', 'question': 'How long have I been using my Fitbit Charge 3?'}, 'N/A'),
        ({'date': '2023/10/27 (Fri) 13:00', 'question': 'How many bikes do I currently own?'}, 'N/A'),
        ({'date': '2023/12/18 (Mon) 04:17',
          'question': 'What was the amount I was pre-approved for when I got my mortgage from Wells Fargo?'},
         'N/A'),
        ({'date': '2023/11/10 (Fri) 04:20',
          'question': 'How many engineers do I lead when I just started my new role as Senior Software Engineer? How many engineers do I lead now?'},
         'N/A'),
    ]

    messages = [{"role": "system", "content": system_prompt}]
    for example_input, example_output in examples:
        messages += [
            {"role": "user", "content": user_prompt.format(example_input['date'], example_input['question'])},
            {"role": "assistant", "content": example_output}
        ]
    messages += [{"role": "user", "content": user_prompt.format(question_date, question)}]
    return messages


def get_contextual_prompt(session: Session, chunk: str) -> list[dict[str, str]]:
    prompt = f"""
<document>

{session.messages}

</document>

<chunk_date>

{session.date}

</chunk_date>

Here is the chunk we want to situate within the whole document

<chunk>

{chunk}

</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else. 
"""
    return [{"role": "user", "content": prompt}]
