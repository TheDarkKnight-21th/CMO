import json
import os
import argparse
import re
from types import SimpleNamespace
from tqdm import tqdm
from vllm import LLM, SamplingParams
import copy
# ==========================================
# 1. 예쁜 JSON 실시간 저장 함수 (필수)
# ==========================================
def append_pretty_json(filepath, new_data):
    # 파일이 없으면 새로 생성
    if not os.path.exists(filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("[\n")
            json.dump(new_data, f, indent=4, ensure_ascii=False)
            f.write("\n]")
        return

    # 파일이 있으면 마지막 ]를 지우고 이어서 씀
    with open(filepath, 'r+', encoding='utf-8') as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell()
        while pos > 0:
            pos -= 1
            f.seek(pos)
            char = f.read(1)
            if char == ']':
                break
        f.seek(pos)
        f.write(",\n")
        json.dump(new_data, f, indent=4, ensure_ascii=False)
        f.write("\n]")

# ==========================================
# 2. Model Wrapper
# ==========================================
class QwenVLLMAdapter:
    def __init__(self, model_name, tensor_parallel_size=1):
        print(f"Loading vLLM model: {model_name} with TP={tensor_parallel_size}...")
        self.llm = LLM(
            model=model_name, 
            tensor_parallel_size=tensor_parallel_size, 
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
            max_model_len=4096,
            enforce_eager=True
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(
            temperature=0.7, 
            top_p=0.95, 
            max_tokens=2048,
            stop=["<|im_end|>", "<|endoftext|>"]
        )

    def __call__(self, messages, **kwargs):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.llm.generate([prompt], self.sampling_params)
        generated_text = outputs[0].outputs[0].text
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=generated_text))])

# ==========================================
# 3. 프롬프트 생성 (검증형 질문 로직)
# ==========================================
def create_verification_prompt(config_item, sentence_item):
    config_str = json.dumps(config_item, indent=2, ensure_ascii=False)
    sentence_str = sentence_item.get("sentence", "")
    
    skills = sentence_item.get("skills", [])
    categories = sentence_item.get("categories", [])
    
    target_list_str = ""
    for i, (skill, cat) in enumerate(zip(skills, categories)):
        target_list_str += f"{i+1}. Skill: '{skill}' (Category: '{cat}')\n"

    system_prompt = (
        "You are an AI assistant specialized in verifying visual attributes. "
        "Your goal is to generate specific verification questions for each skill listed."
    )

    user_prompt = f"""
### Image Metadata (JSON):
{config_str}

### Caption:
"{sentence_str}"

### Target Skills List:
{target_list_str}

### Task:
Generate exactly {len(skills)} questions. Each question must correspond 1-to-1 with the list above.
- If category is 'object', ask specifically about the existence or presence of that object.
- If category is 'texture/shape/color/number', ask about that specific detail (e.g., 'Is the car red?' or 'Are there 2 dogs?').
- If category is 'spatial', ask about the relationship described (e.g., 'Is the cat on the left of the dog?').
- If category is 'style', ask about the image style. (e.g., 'Is this a watercolor painting?').
- If category is 'size', compare the object associated with this size against other objects in the metadata.

**Output Format:**
Provide ONLY a valid JSON list of strings. Do not include any other text.
Example: ["Question 1", "Question 2", "Question 3"]
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

# ==========================================
# 4. JSON 파싱 헬퍼
# ==========================================
def parse_json_response(response_text, expected_length):
    try:
        cleaned_text = re.sub(r'```json\s*|\s*```', '', response_text).strip()
        start = cleaned_text.find('[')
        end = cleaned_text.rfind(']') + 1
        if start != -1 and end != -1:
            cleaned_text = cleaned_text[start:end]
        questions = json.loads(cleaned_text)
        return questions if isinstance(questions, list) else ["Error: Output format is not a list"]
    except:
        return [line.strip() for line in response_text.split('\n') if '?' in line]

# ==========================================
# 5. 메인 실행 로직
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.json (Reference)")
    parser.add_argument("--sentence", type=str, required=True, help="Path to sentence.json (Base)")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tp", type=int, default=1)
    args = parser.parse_args()

    print("Loading datasets...")
    # 1. Config 로드 (참조용 Map 생성)
    with open(args.config, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    config_map = {item['index']: item for item in config_data}

    # 2. Sentence 로드 (순회용)
    with open(args.sentence, 'r', encoding='utf-8') as f:
        sentence_data = json.load(f)

    # Resume 기능
    processed_indices = set()
    if os.path.exists(args.output):
        try:
            with open(args.output, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                for item in existing_data:
                    if 'index' in item:
                        processed_indices.add(item['index'])
            print(f"Resuming... Found {len(processed_indices)} items already processed.")
        except json.JSONDecodeError:
            print("Warning: Output file exists but is invalid. Starting fresh.")

    model = QwenVLLMAdapter(model_name=args.model, tensor_parallel_size=args.tp)

    print("Starting generation...")
    for sentence_item in tqdm(sentence_data):
        idx = sentence_item['index']
        
        if idx in processed_indices:
            continue

        # Config 정보 가져오기 (없으면 경고만 하고 빈 질문 처리)
        config_item = config_map.get(idx)
        if not config_item:
            print(f"Warning: No config for index {idx}")
            sentence_item['questions'] = []
            append_pretty_json(args.output, sentence_item)
            continue

        # ★★★ 핵심 변경: Sentence를 베이스로 사용 ★★★
        # Config의 내용을 복사해오는 게 아니라, Sentence 아이템을 복사해서 씁니다.
        final_item = copy.deepcopy(sentence_item)
        
        # 질문 생성을 위해 LLM에게는 config 정보를 넘겨줌
        messages = create_verification_prompt(config_item, sentence_item)

        try:
            response = model(messages)
            content = response.choices[0].message.content
            skills_len = len(sentence_item.get('skills', []))
            questions_list = parse_json_response(content, expected_length=skills_len)
            
            # Sentence 구조에 'questions' 키만 추가!
            final_item['questions'] = questions_list
            
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            final_item['questions'] = ["Error"]

        # 저장
        append_pretty_json(args.output, final_item)

    print(f"All Done! Saved to {args.output}")

if __name__ == "__main__":
    main()