
# nox
Efficient fine-tuning for ko-llm models

![nox](assets/logo.jpeg)

## News or Update
### 2024.03.15
- upstage/open-ko-llm-leaderboard 1위 (2024/03/15) https://huggingface.co/spaces/upstage/open-ko-llm-leaderboard
![leaderboard](assets/leaderboard.png)

## Hardware and Software
- nvidia driver : 535.86.10
- CUDA Version: 12.2

## Released Model Checkpoints
### [davidkim205/nox-solar-10.7b-v4](https://huggingface.co/davidkim205/nox-solar-10.7b-v4)
| Model                          | Average | Ko-ARC | Ko-HellaSwag | Ko-MMLU | Ko-TruthfulQA | Ko-CommonGen V2 |
| ------------------------------ | ------- | ------ | ------------ | ------- | ------------- | --------------- |
| davidkim205/nox-solar-10.7b-v4 | 67.77   | 73.55  | 72.07        | 57.93   | 79.32         | 55.96           |
### [davidkim205/nox-solar-10.7b-v2](https://huggingface.co/davidkim205/nox-solar-10.7b-v2)
| Model                          | Average | Ko-ARC | Ko-HellaSwag | Ko-MMLU | Ko-TruthfulQA | Ko-CommonGen V2 |
| ------------------------------ | ------- | ------ | ------------ | ------- | ------------- | --------------- |
| davidkim205/nox-solar-10.7b-v2 | 65.38   | 73.46  | 67.32        | 58.7    | 71.94         | 55.49           |
## installation
```
conda create -n nox python=3.10
conda activate nox
pip install -r requirements.txt
```

## Train

### Supervised Fine-Tuning

### DPO Trainer

## Evaluation
https://github.com/davidkim205/kollm_evaluation


## etc
sft와 dpo 학습은 SOLAR 모델에 기준으로 작성되었기때문에 다른 모델을 사용시 template를 변경해야합니다. preprocess.py의 get_template를 참고하세요.
```
def get_template(user, input='', gpt=''):
    if len(input) >= 1:
        return f"### User:\n{user}\n{input}\n### Assistant:{gpt}\n"
    else:
        return f"### User:\n{user}\n### Assistant:\n"

```

# References
- https://github.com/hiyouga/LLaMA-Factory
- https://github.com/OpenAccess-AI-Collective/axolotl
- https://huggingface.co/blog/dpo-trl


