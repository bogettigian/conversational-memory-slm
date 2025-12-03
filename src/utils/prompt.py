def get_prompt(question: str, chunks: list[str]) -> list[dict[str, str]]:
    if chunks:
        previous_conversations = "\n".join([f'\n### Chat {i+1}\n\n"""\n{conversation}\n"""' for i, conversation in enumerate(chunks)])
    else:
        previous_conversations = "No relevant chats history."
    prompt = f"""# Role

You are a helpful assistant that answers the user's question.

# Task

I will give you several history chats between you and a user. Please answer the question based on the relevant chat history. Answer the question step by step: first extract all the relevant information, and then reason over the information to get the answer.

# Chats History
{previous_conversations}

# Question

"{question}"

# Answer (step by step)

"""
    return [{"role": "user", "content": prompt}]