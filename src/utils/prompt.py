def get_prompt(question: str, chunks: list[str]) -> list[dict[str, str]]:
    if chunks:
        previous_conversations = "\n".join([f'\n#### Conversation {i+1}\n\n"""\n{conversation}\n"""' for i, conversation in enumerate(chunks)])
    else:
        previous_conversations = "No previous conversations."
    prompt = f"""## Role

You are a helpful assistant that answers the user's question.

## Task

You are given a question and snippets of relevant previous conversations with the user, if any. You must answer the question.

You MUST answer "I don't know" if it's not possible to answer the question based on the given conversations.

## Inputs

### Question

"{question}"

### Previous Conversations
{previous_conversations}
    """
    return [{"role": "user", "content": prompt}]