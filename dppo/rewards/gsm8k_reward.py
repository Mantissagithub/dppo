from dppo.utils.extract_answer import extract_final_answer


def score_completion(text, target):
    pred = extract_final_answer(text)
    if pred is None:
        return {"reward": 0.0, "is_correct": 0.0, "is_parseable": 0.0, "prediction": None}
    reward = 0.1
    correct = 0.0
    if target is not None and pred == target:
        reward = 1.0
        correct = 1.0
    return {"reward": reward, "is_correct": correct, "is_parseable": 1.0, "prediction": pred}
