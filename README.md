
## Installation

```bash
git clone https://github.com/katiekang1998/llm_hallucinations.git
cd llm_hallucinantions
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

Install FActScore according to instructions [here](https://github.com/shmsw25/FActScore). 
Replace `“InstructGPT”` with `“ChatGPT”` in line 26 of `factscore/atomic_facts.py` in your FActScore installation, since text-davinci-003 is no longer supported. 


## Data

Download training data, and our finetuned checkpoints here.



## Training

See `llm_hallucinations/examples` for training, evaluation, and plotting code. 