from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import os
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, HTTPException
import uvicorn


class ConversationRequest(BaseModel):
    conversation: str

class ScoreResponse(BaseModel):
    score: float

app = FastAPI()

class Verifier:
    def __init__(self):
        model_name = "peiyi9979/math-shepherd-mistral-7b-prm"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).eval()
        self.good_token = '+'
        self.bad_token = '-'
        self.step_tag = 'ки'
        self.candidate_tokens = self.tokenizer.encode(f"{self.good_token} {self.bad_token}")[1:]
        self.step_tag_id = self.tokenizer.encode(f"{self.step_tag}")[-1]

    def score_step(self, conversation):
        input_id = torch.tensor([self.tokenizer.encode(conversation)])

        with torch.no_grad():
            logits = self.model(input_id).logits[:, :, self.candidate_tokens]
            scores = logits.softmax(dim=-1)[:, :, 0]
            step_scores = scores[input_id == self.step_tag_id]
            score = step_scores[0].item()

        return score

    def score_steps_batched(self, conversations):
        inputs_for_prm = self.tokenizer(conversations, return_tensors="pt", padding=True, truncation=True, max_length=4000)

        with torch.no_grad():
            logits = self.model(**inputs_for_prm).logits[:, :, self.candidate_tokens]
            scores_batch = logits.softmax(dim=-1)[:, :, 0]
            scores_batch = scores_batch[inputs_for_prm.input_ids == self.step_tag_id]
            scores_batch = scores_batch.tolist()

        return scores_batch



def select_best_step(steps_batch, scores_batch):
    best_score = max(scores_batch)
    best_index = scores_batch.index(best_score)
    best_step = steps_batch[best_index]
    return best_step, best_score


@app.post("/score_step", response_model=ScoreResponse)
def score_step(request: ConversationRequest):
    try:
        score = verifier.score_step(request.conversation, request.current_step)
        return ScoreResponse(score=score)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def serve():
    uvicorn.run(app, host="0.0.0.0", port=8000)

def debug():
    conversation = """How many days are there in July?"""
    steps = ["Step 1: July has 30 days. So the answer is 31. ки", "Step 1: July has 31 days. So the answer is 31. ки"]
    conversations = [conversation + "\n\n" + step for step in steps]
    # scores = [verifier.score_step(conversation) for conversation in conversations]
    scores = verifier.score_steps_batched(conversations)
    print(scores)

if __name__ == "__main__":
    global verifier
    verifier = Verifier()
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    debug()