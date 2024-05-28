# LLM Hallucinantions

Code for reproducing the experiments in [Unfamiliar Finetuning Examples Control How Language Models Hallucinate](https://arxiv.org/abs/2403.05612). 

## Setup

```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
pip install -e .
```
Install FActScore according to instructions [here](https://github.com/shmsw25/FActScore). 
Replace `“InstructGPT”` with `“ChatGPT”` in line 26 of `factscore/atomic_facts.py` in your FActScore installation, since text-davinci-003 is no longer supported. 

Download the training data and our finetuned checkpoints [here](https://drive.google.com/drive/folders/1dJlwgG6zv8gTezYqeTt8mOst5lEzKo7T?usp=sharing).
Place the downloaded folder inside `llm_hallucinations/examples`. 

## Usage

See `llm_hallucinations/examples` for training, evaluation, and plotting code. 

## Acknowledgements

Our codebase was built on top of [trlx](https://github.com/CarperAI/trlx). 