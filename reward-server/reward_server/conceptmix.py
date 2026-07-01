import os 
# os.environ['HF_HOME']="/data2/hg_models"
import torch
from PIL import Image
import sys

# VLM Import
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("Error: 'vllm' library missing. Please install it.")

def load_qwen_reward(model_path="OpenGVLab/InternVL3_5-8B-Instruct", device="cuda"):
    """
    복잡한 로직 없이 오직 VLM의 Yes/No 대답만으로 Reward를 계산하는 함수.
    """
    
    # ------------------------------------------------------------------
    # 1. 모델 로딩 (초기화)
    # ------------------------------------------------------------------
    if not torch.cuda.is_available():
        device = "cpu"
    
    print(f"Set Device: {device}")
    print(f"[Init] Loading Qwen-VL: {model_path}...")

    try:
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.26, 
            tensor_parallel_size=1,
            max_model_len=4096,
            limit_mm_per_prompt={"image": 1},
            enforce_eager=True
        )
        
        tokenizer = llm.get_tokenizer()
        
        # 'Yes' 또는 'No'만 나오게 유도
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=10,
            stop=["<|im_end|>", "\n", "."]
        )
        print(" -> Qwen-VL Ready.")

    except Exception as e:
        print(f" -> Error loading Qwen-VL: {e}")
        return None

    # ------------------------------------------------------------------
    # 2. 계산 함수 (Closure)
    # ------------------------------------------------------------------
    def compute_ours(image: Image.Image, refined_config: dict):
        """
        Args:
            image: 생성된 이미지
            refined_config: 'questions' 리스트가 포함된 딕셔너리
            
        Returns:
            reward_tensor: (1, N) 형태의 텐서 (모든 점수 플랫하게)
            detailed_object_rewards: 로그용 딕셔너리
            detailed_spatial_scores: 로그용 리스트
        """
        

        # 1. 질문 가져오기
        questions = refined_config.get("questions", [])
        # print(f"Questions : {questions}")
        # 질문이 없으면 0점 리턴
        if not questions:
            return torch.tensor([[0.0]], device=device), {}, []

        # 2. 배치 프롬프트 생성
        prompts = []

        for q in questions:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"{q}\nAnswer simply with Yes or No."}
                    ]
                }
            ]

            prompt_text = tokenizer.apply_chat_template(
                conversation, 
                tokenize=False, 
                # add_generation_prompt=True,

            )
            # print(prompt_text, q)
            # exit()
            prompts.append({
                "prompt": prompt_text,
                "multi_modal_data": {"image": image}
            })

        # 3. VLM 추론 (한방에)
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        # 4. 결과 파싱 (조건문 없이 무조건 Yes면 1.0)
        flat_rewards_list = []
        
        # 리턴 포맷을 맞추기 위한 껍데기 변수들
        detailed_object_rewards = {}
        detailed_spatial_scores = []
        
        skills = refined_config.get("skills", [])
        categories = refined_config.get("categories", [])

        for i, output in enumerate(outputs):
            # 텍스트 추출
            ans = output.outputs[0].text.strip().lower()
            # print(f"  [Q{i}] Answer: {ans}")
            
            # ★ 핵심: 조건 없이 Yes면 1점, 아니면 0점 ★
            score = 1.0 if "yes" in ans else 0.0
            
            flat_rewards_list.append(score)

            # --- 로그 정리 (Training Loop 호환용) ---
            # 카테고리가 뭐든 채점 방식은 똑같지만, 리턴 포맷(딕셔너리/리스트)은 
            # 호출하는 쪽에서 기대하는 모양대로 나눠 담아줍니다.
            
            # 메타데이터 안전하게 가져오기
            cat = categories[i] if i < len(categories) else "unknown"
            skill_name = skills[i] if i < len(skills) else f"q_{i}"

            if cat == "spatial":
                detailed_spatial_scores.append(score)
            else:
                # 객체 관련 점수는 딕셔너리에 기록
                # (키가 겹치지 않게 인덱스를 붙여서 저장)
                obj_key = f"{i}_{skill_name}"
                if obj_key not in detailed_object_rewards:
                    detailed_object_rewards[obj_key] = {}
                detailed_object_rewards[obj_key][cat] = score

        # 5. 텐서 변환 및 리턴
        reward_tensor = torch.tensor([flat_rewards_list], dtype=torch.float32, device=device)
        
        return reward_tensor, detailed_object_rewards, detailed_spatial_scores

    return compute_ours