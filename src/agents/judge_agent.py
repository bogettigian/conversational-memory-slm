from litellm import completion

from src.datasets.dataset import LongMemEvalInstance


class JudgeAgent:
    def __init__(self, judge_model_name: str):
        self.judge_model_name = judge_model_name

    def judge(self, instance: LongMemEvalInstance, predicted_answer):
        prompt = f"""
        You are a helpful assistant that judges the correctness of an answer to a question.
        The question is: {instance.question}
        The memory agent answer is: {predicted_answer}
        The ground truth answer is: {instance.answer}
        Return True if the prediction is correct, False otherwise. No other text or explanation.
        """
        messages = [{"role": "user", "content": prompt}]
        response = completion(model=self.judge_model_name, messages=messages)
        judgment = response.choices[0].message.content
        return judgment.lower() == "true"
