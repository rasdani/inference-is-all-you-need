from vllm import LLM, SamplingParams
import re
import requests
import json
from datetime import datetime
import os


class Solver:
    def __init__(self, model_name="Qwen/Qwen2-Math-1.5B-Instruct"):
        self.solver_llm = LLM(model=model_name, tensor_parallel_size=2, quantization="fp8")
        self.sampling_params = SamplingParams(
            temperature=0.7, 
            top_p=0.8, 
            repetition_penalty=1.05, 
            max_tokens=512, 
            # max_tokens=2048, 
            # stop=["Step"]
            stop=["\n\n"]
        )

    def generate_step(self, conversation, num_outputs_per_step):
        outputs = self.solver_llm.generate([conversation] * num_outputs_per_step, self.sampling_params)
        return outputs


def format_conversation_for_prm(conversation_string, problem_string):
    # breakpoint()
    steps_body = conversation_string.split("Let's think step by step.\n\n")[1]
    steps = steps_body.split("\n\n")
    steps = [step for step in steps if step not in ["", " "]]

    # Process each step
    formatted_steps = []
    for step in steps[:-1]:
        formatted_step = step.replace("\n", " ").strip()
        formatted_step = formatted_step.replace("  ", " ")
        # add good tag to every line so far
        formatted_step += f" +"
        formatted_steps.append(formatted_step)


    # add step tag for PRM at the end of the last step
    last_step = steps[-1].replace("\n", " ").strip()
    last_step += " ки"
    formatted_steps.append(last_step)
    # Combine the problem and formatted steps
    result = problem_string + "\n" + "\n\n".join(formatted_steps)

    return result


def score_steps_batched(conversations):
    url = "http://localhost:8000/score_steps_batched"
    response = requests.post(url, json={"conversations": conversations})
    if response.status_code == 200:
        return response.json()["scores"]
    else:
        raise Exception(f"Error scoring steps: {response.text}")



def select_best_step(steps_batch, scores_batch):
    best_score = max(scores_batch)
    best_index = scores_batch.index(best_score)
    # best_step = steps_batch[best_index]
    return best_index, best_score


def solve_problem(problem, num_outputs_per_step=2):
    chat_template = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
    "<|im_start|>assistant\n"
    )
    prefix = "Let's think step by step.\n\nStep 1: "
    convo_start = chat_template.format(instruction=problem) + prefix

    results = {"conversation": convo_start, "formatted_conversation": [], "steps": [], "scores": []}

    max_num_steps = 15
    for _ in range(max_num_steps):
        outputs = solver.generate_step(results["conversation"], num_outputs_per_step=num_outputs_per_step)
        output_texts = [output.outputs[0].text.strip() for output in outputs]
        
        if all(output_text == "" for output_text in output_texts):
            continue


        formatted_conversations = []
        for output_text in output_texts:
            current_conversation = results["conversation"] + output_text.strip()
            formatted_conversation = format_conversation_for_prm(current_conversation, problem)
            formatted_conversations.append(formatted_conversation)

        scores = score_steps_batched(formatted_conversations)
        best_index, best_score = select_best_step(output_texts, scores)

        best_output_text = output_texts[best_index].strip()
        results["conversation"] += best_output_text
        results["formatted_conversation"].append(formatted_conversations[best_index])
        results["steps"].append(best_output_text)
        results["scores"].append(best_score)

        if "\\boxed{" in formatted_conversations[best_index]:
            # final_answer = re.search(r"\\boxed\{(.*?)\}", formatted_conversations[best_index])
            final_answer = re.search(r"\\boxed\{((?:[^{}]|{(?:[^{}]|{[^{}]*})*})*)\}", formatted_conversations[best_index])
            if final_answer:
                results["final_answer"] = final_answer.group(1)
            else:
                results["final_answer"] = ""
            # breakpoint()
            break

        results["conversation"] += "\n\n"
        results["final_answer"] = ""

    return results


def main():
    global solver
    solver = Solver()

    samples_path = "./test.jsonl"
    print(f"Reading {samples_path}")
    # n = 10
    # n = 50
    n = 30
    # n = 100
    # n = 30
    # n = None
    with open(samples_path) as f:
        samples = [json.loads(l) for l in f.readlines() if l]
        samples = samples[:n]

    model_name = "Qwen2-Math_1_5B-Instruct"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"results_{model_name}_{timestamp}.jsonl"
    os.makedirs("results", exist_ok=True)

    num_outputs_per_step = 2
    results = []
    for i, sample in enumerate(samples):
        print(f"SOLVING PROBLEM {i+1}/{len(samples)}:\n{sample['problem']}")
        problem = sample["problem"]
        solution = solve_problem(problem, num_outputs_per_step=num_outputs_per_step)
        conversation = solution["conversation"]
        formatted_conversation = solution["formatted_conversation"]
        scores = solution["scores"]
        steps = solution["steps"]
        answer = solution["final_answer"]
        print(f"FINAL ANSWER: {answer}")
        print(f"GROUND TRUTH: {sample['answer']}")
        results.append({
            **sample,
            "conversation": conversation,
            "formatted_conversation": formatted_conversation,
            "scores": scores,
            "steps": steps,
            "model_answer": answer
        })

        # Overwrite to save intermediate results
        with open(f"results/{results_filename}", "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()

