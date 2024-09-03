import json
# from grading.grader import grade_answer
from grader import grade_answer
from fire import Fire


def evaluate_results(results_file: str):
    correct = 0
    total = 0

    with open(results_file, 'r') as f:
        for line in f:
            result = json.loads(line)
            # model_answer = result['answer']
            # gt_answer = result['gt_answer']
            model_answer = result['model_answer']
            gt_answer = result['answer']
            
            is_correct = grade_answer(model_answer, gt_answer)
            
            if is_correct:
                correct += 1
            total += 1

            print(f"Problem: {result['problem']}")
            # print(f"Raw Response: {result['raw_response']}")
            print(f"Model Answer: {model_answer}")
            print(f"Ground Truth: {gt_answer}")
            print(f"Correct: {is_correct}")
            print("-" * 50)

    accuracy = correct / total if total > 0 else 0
    print(f"\nFinal Accuracy: {accuracy:.2%} ({correct}/{total})")

if __name__ == "__main__":
    Fire(evaluate_results)