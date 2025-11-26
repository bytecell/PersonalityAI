#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, torch, argparse, pandas as pd, numpy as np
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, GenerationConfig
from peft import LoraConfig
from trl import SFTTrainer
from sklearn.metrics import mean_absolute_error

def test_compute_metrics(label, pred):
    MAE = []
    for i in range(5):
        if len(pred) <= i:
            raise ValueError("prediction length too short")
        s = pred[i]
        if isinstance(s, str) and len(s) > 0 and s[-1] == '.':
            s = s[:-1]
        MAE.append(1 - mean_absolute_error([float(label[i])], [float(s)]))
    return MAE

class multi_task:
    def __init__(self):
        self.model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
        self.ocean_path = "/home/visi8747/juphome/2025_KETI_CORPUS_TXT_data/split_output_combined"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def load_ocean_dataset(self, split):
        df = pd.read_csv(os.path.join(self.ocean_path, f"{split}_unstyled.csv"))
        df = df.dropna(subset=['unstyled_text'])
        df['instruction'] = df['unstyled_text']
        df['output'] = df.apply(
            lambda r: f"N={round(r['neuroticism'],2)}, E={round(r['extraversion'],2)}, "
                      f"O={round(r['openness'],2)}, A={round(r['agreeableness'],2)}, "
                      f"C={round(r['conscientiousness'],2)}",
            axis=1
        )
        return Dataset.from_pandas(df[['instruction', 'output']])

    def make_styled_dataset(self, split):
        df = pd.read_csv(os.path.join(self.ocean_path, f"{split}_unstyled.csv"))
        df = df.dropna(subset=['unstyled_text'])
        def extract_first_sentence(t):
            s = re.split(r'(?<=[.!?])\s+', str(t).strip())
            return s[0] if s else str(t)
        def make_inst(r):
            o = f"[N={round(r['neuroticism'],2)}, E={round(r['extraversion'],2)}, " \
                f"O={round(r['openness'],2)}, A={round(r['agreeableness'],2)}, " \
                f"C={round(r['conscientiousness'],2)}]"
            return f"다음 문장의 화자 성격을 기반으로 문장을 생성하시오.\n성격 점수: {o}\n문장: {r['unstyled_text']}"
        df['unstyled_text'] = df['unstyled_text'].apply(extract_first_sentence)
        df['instruction'] = df.apply(make_inst, axis=1)
        df['output'] = df['unstyled_text']
        return Dataset.from_pandas(df[['instruction', 'output']])

    def train_model(self):
        train_ocean = self.load_ocean_dataset("train")
        val_ocean = self.load_ocean_dataset("valid")
        train_styled = self.make_styled_dataset("train")
        train_dataset = concatenate_datasets([train_ocean, train_styled])
        val_dataset = val_ocean
        def format_prompt(e):
            return f"### Instruction: {e['instruction']}\n\n### Response: {e['output']}"
        lora = LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM"
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16
        )
        model.gradient_checkpointing_enable()
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=lora,
            formatting_func=format_prompt,
            args=TrainingArguments(
                output_dir="outputs",
                per_device_train_batch_size=32,
                gradient_accumulation_steps=8,
                eval_strategy="steps",
                eval_steps=100,
                warmup_steps=200,
                logging_steps=100,
                num_train_epochs=2,
                learning_rate=3e-4,
                fp16=True,
                report_to="none",
                load_best_model_at_end=True
            )
        )
        trainer.train()
        model = trainer.model.merge_and_unload()
        model.save_pretrained("Bllossom-8B-ft-multitask")
        self.tokenizer.save_pretrained("Bllossom-8B-ft-multitask")

    def build_prompt(self, mode):
        base = (
            "당신은 문장에 대해 OCEAN 성격 요소 각각에 대한 점수를 알려주는 전문가 입니다.\n"
            "OCEAN 성격 요소는 O(Openness), C(Conscientiousness), E(Extraversion), A(Agreeableness), N(Neuroticism)\n"
            "문장을 읽고 각 요소를 0~1 사이로 점수화하세요.\n"
            "출력 형식: [O: 0.6722, C: 0.8322, E: 0.4860, A: 0.5298, N: 0.9323]\n"
        )
        examples = {
            "one": [("아메리카노랑 이제 카페니까 네", "[O:0.50,C:0.63,E:0.45,A:0.54,N:0.22]")],
            "five": [
                ("저 펭수 2순위로 했어요. 저희", "[O:0.50,C:0.63,E:0.45,A:0.54,N:0.22]"),
                ("대구에는 뭐 많지 대구는 뭐 알지", "[O:0.57,C:0.71,E:0.67,A:0.45,N:0.52]"),
                ("일단 아침 8시에 인천공항에 도착", "[O:0.50,C:0.63,E:0.45,A:0.54,N:0.22]"),
                ("9위가 이수근이었어요.", "[O:0.57,C:0.71,E:0.67,A:0.45,N:0.52]"),
                ("야간으로 놀러 다녀야 될 거 아니에요", "[O:0.57,C:0.71,E:0.67,A:0.45,N:0.52]")
            ],
            "ten": [
                ("그래서 바짝 놀자는 거야", "[O:0.57,C:0.71,E:0.67,A:0.45,N:0.52]"),
                ("극 J 엑셀 사용하죠", "[O:0.50,C:0.63,E:0.45,A:0.54,N:0.22]"),
                ("끝 아 로봇", "[O:0.57,C:0.71,E:0.67,A:0.45,N:0.52]"),
                ("좋은 호텔이라든가 물배요?", "[O:0.57,C:0.71,E:0.67,A:0.45,N:0.52]"),
                ("닭갈비를 먹이고", "[O:0.72,C:0.53,E:0.48,A:0.68,N:0.47]"),
                ("맞습니다 어", "[O:0.50,C:0.63,E:0.45,A:0.54,N:0.22]"),
                ("선재 응", "[O:0.72,C:0.53,E:0.48,A:0.68,N:0.47]"),
                ("용인?", "[O:0.72,C:0.53,E:0.48,A:0.68,N:0.47]"),
                ("푸바오 보고 왔었는데", "[O:0.57,C:0.71,E:0.67,A:0.45,N:0.52]"),
                ("음료수 옆에 있으니까 딱 맞을 것 같아", "[O:0.57,C:0.71,E:0.67,A:0.45,N:0.52]")
            ]
        }
        for s, v in examples.get(mode, []):
            base += f'문장: "{s}", 결과- {v}\n'
        return base

    def infer_and_score(self, mode):
        model = AutoModelForCausalLM.from_pretrained("Bllossom-8B-ft-multitask", torch_dtype=torch.float16).to(self.device)
        tok = AutoTokenizer.from_pretrained("Bllossom-8B-ft-multitask")
        df = pd.read_csv(os.path.join(self.ocean_path, "output_combined_test.csv"))
        df['label'] = df.apply(
            lambda r: [float(r['openness']), float(r['conscientiousness']),
                       float(r['extraversion']), float(r['agreeableness']),
                       float(r['neuroticism'])],
            axis=1
        )
        prefix = self.build_prompt(mode)
        gen_cfg = GenerationConfig(max_new_tokens=50, do_sample=True, min_new_tokens=15, temperature=0.9,
                                   repetition_penalty=1.3, top_k=50, top_p=0.92,
                                   eos_token_id=model.config.eos_token_id)
        p = re.compile(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)")
        score = []
        for text in tqdm(df["script"].to_list(), desc=f"{mode}-shot inference"):
            user = str(text)
            inputs = tok.encode(prefix + f'문장: "{user}", 결과- [O: ', return_tensors="pt").to(self.device)
            outputs = model.generate(inputs, generation_config=gen_cfg)
            new_toks = outputs[0][inputs.shape[-1]:]
            out_text = tok.decode(new_toks, skip_special_tokens=True)
            tmp = p.findall(out_text)
            score.append(tmp)
        MAE_arr = []
        error_idx = []
        for i in range(len(df)):
            try:
                MAE_arr.append(test_compute_metrics(df['label'].iloc[i][:5], score[i][:5]))
            except Exception:
                error_idx.append(i)
        while len(error_idx) != 0:
            re_df = pd.DataFrame(columns=["Text", "Inference"])
            for data in tqdm(df['script'].iloc[error_idx].to_list(), desc=f"{mode}-shot retry"):
                user = str(data)
                inputs = tok.encode(prefix + f'문장: "{user}", 결과- [O: ', return_tensors="pt").to(self.device)
                outputs = model.generate(inputs, generation_config=gen_cfg)
                new_toks = outputs[0][inputs.shape[-1]:]
                out_text = tok.decode(new_toks, skip_special_tokens=True)
                re_df = pd.concat([re_df, pd.DataFrame({'Text': [user], 'Inference': [out_text]})], ignore_index=True)
            re_score = []
            for content in re_df['Inference'].to_list():
                tmp = p.findall(content)
                re_score.append(tmp)
            for i, idx in enumerate(error_idx):
                score[idx] = re_score[i]
            MAE_arr = []
            new_error = []
            for i in range(len(df)):
                try:
                    MAE_arr.append(test_compute_metrics(df['label'].iloc[i][:5], score[i][:5]))
                except Exception:
                    new_error.append(i)
            error_idx = new_error
        factor_mae = np.array(MAE_arr).mean(axis=0)
        avg_mae = np.mean(factor_mae)
        print("factor_mae:", factor_mae)
        print("avg_mae:", avg_mae)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["zero", "one", "five", "ten"], required=True)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    runner = multi_task()
    if args.train:
        runner.train_model()
    runner.infer_and_score(args.mode)

if __name__ == "__main__":
    main()

