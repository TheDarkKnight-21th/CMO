import json

input_file = 'gen.jsonl' # 실제 사용 중인 파일명으로 맞춰주세요.
config_output_file = 'config_output.json'
sentence_output_file = 'sentence_output.json'

config_data = []
sentence_data = []

with open(input_file, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        line = line.strip() # 양옆 공백 및 줄바꿈 제거
        if not line:
            continue
            
        # 🚨 여기에 try-except를 추가하여 에러를 잡아냅니다!
        try:
            data = json.loads(line)
        except json.decoder.JSONDecodeError as e:
            print(f"❌ [문법 에러] {idx + 1}번째 줄에서 JSON 변환 실패!")
            print(f"▶️ 원인: {e}")
            print(f"▶️ 문제의 텍스트: {line}")
            print("-" * 50)
            break # 에러가 나면 즉시 중단합니다. (건너뛰려면 break 대신 continue 사용)
        
        # --- (이하 기존 변환 로직 동일) ---
        objects = []
        relations = []
        skills = []
        categories = []
        
        # style = "photo" if "photo" in data["prompt"].lower() else "unknown"
        # skills.append(style)
        # categories.append("style")
        
        for i, item in enumerate(data.get("include", [])):
            obj_id = i + 1
            
            obj_dict = {
                "id": obj_id,
                "item": item["class"]
            }
            skills.append(item["class"])
            categories.append("object")
            
            if "color" in item:
                obj_dict["color"] = item["color"]
                skills.append(item["color"])
                categories.append("color")
                
            if "count" in item and (item["count"] > 1 or data.get("tag") == "counting"):
                count_str = str(item["count"])
                obj_dict["number"] = count_str
                skills.append(count_str)
                categories.append("number")
                
            objects.append(obj_dict)
            
            if "position" in item:
                pos_name = item["position"][0]
                target_index = item["position"][1]
                target_id = target_index + 1
                
                relation_dict = {
                    "name": pos_name,
                    "description": f"{{ObjectA}} is {pos_name} {{ObjectB}}",
                    "ObjectA_id": str(obj_id),
                    "ObjectB_id": str(target_id)
                }
                relations.append(relation_dict)
                skills.append(pos_name)
                categories.append("spatial")

        config = {
            "objects": objects,
            "relation": relations,
            "index": idx,
            # "style": style
        }
        config_data.append(config)
        
        sentence = {
            "index": idx,
            "sentence": data["prompt"],
            "categories": categories,
            "skills": skills
        }
        sentence_data.append(sentence)

# 파일 저장 로직은 동일
with open(config_output_file, 'w', encoding='utf-8') as f_conf:
    json.dump(config_data, f_conf, indent=4, ensure_ascii=False)

with open(sentence_output_file, 'w', encoding='utf-8') as f_sent:
    json.dump(sentence_data, f_sent, indent=4, ensure_ascii=False)

print(f"✅ 변환 완료!")