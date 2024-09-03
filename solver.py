from vllm import LLM, SamplingParams
import re
import requests


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


if __name__ == "__main__":
    solver = Solver()
    chat_template = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
    "<|im_start|>assistant\n"
    )
    prefix = "Let's think step by step.\n\nStep 1: "
    problem = """In the month of July, the bookstore has a sale on days that are multiples of 5, and the shoe store has a sale every 6 days. If the shoe store has a sale on July 3, how many times in July do both stores have sales on the same date?"""
    convo_start = chat_template.format(instruction=problem) + prefix

    results = {"conversation": convo_start, "steps": [], "scores": []}

    max_num_steps = 15
    for _ in range(max_num_steps):
        outputs = solver.generate_step(results["conversation"], num_outputs_per_step=2)
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
        breakpoint()

        results["conversation"] += output_texts[best_index].strip()
        results["scores"].append(best_score)

        if "\\boxed{" in formatted_conversations[best_index]:
            final_answer = re.search(r"\\boxed\{(.*?)\}", formatted_conversations[best_index])
            if final_answer:
                results["final_answer"] = final_answer.group(1)
            else:
                results["final_answer"] = ""
            breakpoint()
            break

        results["conversation"] += "\n\n"


