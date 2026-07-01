import os
import json
import torch
import numpy as np
from PIL import Image
import logging
import cv2


try:
    from transformers import Sam3Processor, Sam3Model
except ImportError:
    print("Error: 'transformers' library missing or SAM3 not supported.")


try:
    import open_clip as std_open_clip
except ImportError:
    print("Error: 'open_clip' library missing.")


try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    pass


try:
    from hpsv2.src.open_clip import create_model_and_transforms as hps_create_model
    from hpsv2.src.open_clip import get_tokenizer as hps_get_tokenizer
except ImportError:
    print("Warning: 'hpsv2' module not found. Make sure hpsv2 is in PYTHONPATH.")


logging.getLogger("depth_anything_3.api").setLevel(logging.ERROR)


def load_ours(attribute_dir="/data1/jungmyungwi/ICML2026/DanceGRPO/assets/conceptmix_config", device="cuda"):

    """
    Initializes SAM3, OpenCLIP, DepthAnything, and HPSv2.
    Returns the `compute_ours` closure function.
    """
    



    if not torch.cuda.is_available():
        device = "cpu"
    print(f"Set Device: {device}")
    print("=" * 50)


    print("[1/4] Loading SAM3...")
    sam_processor = None
    sam_model = None
    try:
        sam_processor = Sam3Processor.from_pretrained("facebook/sam3")
        sam_model = Sam3Model.from_pretrained("facebook/sam3").to(device)
        sam_model.eval()
    except Exception as e:
        print(f" -> Warning: Failed to load SAM3. {e}")


    print("[2/4] Loading Standard OpenCLIP (ViT-H-14)...")
    clip_model = None
    clip_preprocess = None
    clip_tokenizer = None
    try:
        model_name, pretrained = 'ViT-H-14', 'laion2b_s32b_b79k'
        clip_model, _, clip_preprocess = std_open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        clip_tokenizer = std_open_clip.get_tokenizer(model_name)
        clip_model.eval()
    except Exception as e:
        print(f" -> Warning: Failed to load Standard OpenCLIP. {e}")


    print("[3/4] Loading Depth Anything V3...")
    depth_model = None
    try:
        depth_model = DepthAnything3.from_pretrained("depth-anything/da3-small").to(device)
        depth_model.eval()
    except Exception as e:
        print(f" -> Warning: Failed to load DepthAnything3. {e}")


    print("[4/4] Loading HPSv2 (Reward Model)...")
    hps_model = None
    hps_preprocess = None
    hps_tokenizer = None
    try:

        model, _, preprocess_val = hps_create_model(
            'ViT-H-14',
            '/data1/jungmyungwi/ICML2026/DanceGRPO/hps_ckpt/open_clip_pytorch_model.bin',
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )
        

        cp_path = "/data1/jungmyungwi/ICML2026/DanceGRPO/hps_ckpt/HPS_v2.1_compressed.pt"
        if os.path.exists(cp_path):
            checkpoint = torch.load(cp_path, map_location=device)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            
            hps_model = model
            hps_preprocess = preprocess_val
            hps_tokenizer = hps_get_tokenizer('ViT-H-14')
            print(" -> HPSv2 Loaded Successfully.")
        else:
            print(f" -> Warning: HPSv2 checkpoint not found at {cp_path}")

    except Exception as e:
        print(f" -> Warning: Failed to load HPSv2. {e}")

    print("=" * 50)




    visual_keys = ["texture", "shape", "style", "color"]
    
    def _load_all_json_files():
        configs = {}
        if not os.path.exists(attribute_dir): return configs
        for key in visual_keys:
            path = os.path.join(attribute_dir, f"{key}.json")
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    configs[key] = json.load(f)
        return configs

    attribute_configs = _load_all_json_files()
    cached_text_features = {}
    cached_labels = {}

    template_map = {
        "texture": ["{object} made of {skill}.", "{object} with {skill} texture."],
        "color": ["a {skill} {object}.", "a {skill}-colored {object}."],
        "shape": ["a {skill}-shaped {object}.", "{object} in the shape of {skill}."],
        "style": ["{object} in {skill} style.", "a {skill} painting of {object}."],
        "default": ["a photo of a {skill} {object}.", "a {skill} {object}."]
    }
    
    print(">>> System Initialized (load_ours). ready to compute.")






    def _get_objects_safe(detected_objs, raw_id):
        if raw_id is None: return []
        if raw_id in detected_objs: return detected_objs[raw_id]
        try:
            if int(raw_id) in detected_objs: return detected_objs[int(raw_id)]
        except: pass
        try:
            if str(raw_id) in detected_objs: return detected_objs[str(raw_id)]
        except: pass
        return []


    @torch.no_grad()
    def get_sam_result(image, prompt_text):
        if sam_processor is None or sam_model is None: return {'masks': [], 'boxes': [], 'scores': []}
        inputs = sam_processor(images=image, text=prompt_text, return_tensors="pt").to(device)
        outputs = sam_model(**inputs)
        results = sam_processor.post_process_instance_segmentation(
            outputs, threshold=0.5, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        return results


    @torch.no_grad()
    def get_depth_map(image):
        if depth_model is None: return None
        try:
            pred = depth_model.inference([np.array(image)])
            depth = pred.depth[0]
            if isinstance(depth, torch.Tensor): depth = depth.cpu().numpy()
            
            if depth.shape[:2] != image.size[::-1]:
                 d_img = Image.fromarray(depth)
                 d_img = d_img.resize(image.size, resample=Image.BILINEAR)
                 depth = np.array(d_img)
            
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min > 0: depth = (depth - d_min) / (d_max - d_min)
            return depth
        except: return None


    @torch.no_grad()
    def calculate_visual_reward(image, mask, box, attr_key, target_val, item_name, use_bbox_only=True):
        cache_key = (attr_key, item_name)
        

        if cache_key not in cached_text_features:
            if attr_key not in attribute_configs: return 0.0
            data = attribute_configs[attr_key]
            skills = list(data.get("skill", {}).keys())
            templates = template_map.get(attr_key, template_map["default"])
            
            feats_list = []
            for skill in skills:
                prompts = [t.format(skill=skill, object=item_name) for t in templates]
                tokens = clip_tokenizer(prompts).to(device)
                f = clip_model.encode_text(tokens)
                f /= f.norm(dim=-1, keepdim=True)
                mean_f = f.mean(dim=0, keepdim=True)
                mean_f /= mean_f.norm(dim=-1, keepdim=True)
                feats_list.append(mean_f)
            
            cached_text_features[cache_key] = torch.cat(feats_list, dim=0)
            cached_labels[cache_key] = skills

        text_feats = cached_text_features[cache_key]
        labels = cached_labels[cache_key]
        if target_val not in labels: return 0.0
        target_idx = labels.index(target_val)

        if attr_key == "style":
            inp = clip_preprocess(image).unsqueeze(0).to(device)
        else:

            x1, y1, x2, y2 = map(int, box)
            if use_bbox_only:

                crop = image.crop((x1, y1, x2, y2))
            else:
                img_arr = np.array(image)
                bg_color = np.array([240, 240, 240], dtype=np.uint8)
                mask_3d = mask[:, :, None]
                masked_img_arr = np.where(mask_3d, img_arr, bg_color)
                masked_pil = Image.fromarray(masked_img_arr.astype(np.uint8))
                
                x1, y1, x2, y2 = map(int, box)
                crop = masked_pil.crop((x1, y1, x2, y2))
            inp = clip_preprocess(crop).unsqueeze(0).to(device)
        
        img_f = clip_model.encode_image(inp)
        img_f /= img_f.norm(dim=-1, keepdim=True)
        
        logits = (img_f @ text_feats.T) * clip_model.logit_scale.exp()
        probs = torch.nn.functional.softmax(logits, dim=-1)
        reward = probs[0, target_idx].item()
        return reward


    def calculate_number_reward(count, target):
        try: t = int(target)
        except: return 0.0
        return 1.0 / ((abs(count - t) + 1.0) ** 2)

    def calculate_size_reward(area, all_areas, target):
        valid = [a for a in all_areas if a > 0]
        if area not in valid: return 0.0
        target = target.lower()
        if target == "huge": sorted_areas = sorted(valid, reverse=True)
        elif target == "tiny": sorted_areas = sorted(valid)
        else: return 0.0
        return 1.0 / ((sorted_areas.index(area) + 1.0) ** 2)


    def _check_single_pair(info_A, info_B, depth_map, relation_name):
        box_A, box_B = info_A['box'], info_B['box']
        area_A, area_B = info_A['area'], info_B['area']
        
        cx_A, cy_A = (box_A[0] + box_A[2]) / 2, (box_A[1] + box_A[3]) / 2
        cx_B, cy_B = (box_B[0] + box_B[2]) / 2, (box_B[1] + box_B[3]) / 2

        x_inter_1 = max(box_A[0], box_B[0])
        x_inter_2 = min(box_A[2], box_B[2])
        x_overlap_len = max(0, x_inter_2 - x_inter_1)
        
        y_inter_1 = max(box_A[1], box_B[1])
        y_inter_2 = min(box_A[3], box_B[3])
        y_overlap_len = max(0, y_inter_2 - y_inter_1)

        w_min = min(box_A[2]-box_A[0], box_B[2]-box_B[0])
        h_min = min(box_A[3]-box_A[1], box_B[3]-box_B[1])
        tol_x = w_min * 0.1
        tol_y = h_min * 0.1
        
        relation_name = relation_name.lower().strip()

        if relation_name == "left":
            is_strictly_left = (box_A[2] < box_B[0] + tol_x)
            is_y_aligned = (y_overlap_len > 0)
            if is_strictly_left and is_y_aligned: return 1.0
            if cx_A < cx_B: return 0.5
            return 0.0

        elif relation_name == "right":
            is_strictly_right = (box_A[0] > box_B[2] - tol_x)
            is_y_aligned = (y_overlap_len > 0)
            if is_strictly_right and is_y_aligned: return 1.0
            if cx_A > cx_B: return 0.5
            return 0.0

        elif relation_name in ["top", "above", "on top of"]:
            is_strictly_above = (box_A[3] < box_B[1] + tol_y)
            is_x_aligned = (x_overlap_len > 0)
            if is_strictly_above and is_x_aligned: return 1.0
            if cy_A < cy_B: return 0.5
            return 0.0

        elif relation_name in ["bottom", "below", "under"]:
            is_strictly_below = (box_A[1] > box_B[3] - tol_y)
            is_x_aligned = (x_overlap_len > 0)
            if is_strictly_below and is_x_aligned: return 1.0
            if cy_A > cy_B: return 0.5
            return 0.0

        elif relation_name == "inside":
            inter_area = x_overlap_len * y_overlap_len
            if area_A > 0 and (inter_area / area_A) >= 0.95: return 1.0
            return 0.0

        elif relation_name == "outside":
            inter_area = x_overlap_len * y_overlap_len
            min_area = min(area_A, area_B)
            if min_area > 0 and (inter_area / min_area < 0.1): return 1.0
            return 0.0
        elif relation_name in ["in front of", "behind"] and depth_map is not None:

            mask_A = info_A['mask']
            mask_B = info_B['mask']


            depth_vals_A = depth_map[mask_A.astype(bool)]
            depth_vals_B = depth_map[mask_B.astype(bool)]

            score_A = np.mean(depth_vals_A) if depth_vals_A.size > 0 else 1.0
            score_B = np.mean(depth_vals_B) if depth_vals_B.size > 0 else 1.0



            if relation_name == "in front of":

                if score_A < score_B - 0.02: return 1.0
                if score_A < score_B: return 0.5 
                return 0.0
            
            elif relation_name == "behind":

                if score_A > score_B + 0.02: return 1.0
                if score_A > score_B: return 0.5
                return 0.0
        
        return 0.0

    def calculate_relation_reward(list_A, list_B, depth_map, relation_name):
        if not list_A or not list_B: return 0.0
        total_score_sum = 0.0
        for obj_A in list_A:
            best_score_for_this_A = 0.0
            for obj_B in list_B:
                score = _check_single_pair(obj_A, obj_B, depth_map, relation_name)
                if score > best_score_for_this_A:
                    best_score_for_this_A = score
                if best_score_for_this_A >= 1.0: break
            total_score_sum += best_score_for_this_A
        return total_score_sum / len(list_A)




    def compute_ours(image: Image.Image, refined_config: dict, prompt=None):
        """
        Calculates rewards based on Object Detection, Spatial Relations, and HPSv2.
        
        Args:
            image (PIL.Image): The generated image.
            refined_config (dict): Contains 'objects', 'relation', 'style', and 'prompt'.
            prompts (list, optional): List of prompt strings.

        Returns:
            reward_tensor (torch.Tensor): Shape (1, N)
            detailed_object_rewards (dict): Logs
            detailed_spatial_scores (list): Logs
        """
        

        detected_objs = {}
        all_areas = []
        for obj in refined_config.get("objects", []):
            try: target_cnt = int(obj.get("number", 99))
            except: target_cnt = 99
            
            res = get_sam_result(image, obj["item"])
            found = []
            if len(res['masks']) > 0:
                indices = torch.argsort(res['scores'], descending=True)
                num_take = min(len(indices), target_cnt)
                
                for i in range(num_take):
                    idx = indices[i]
                    box = res['boxes'][idx].cpu().numpy()
                    mask = res['masks'][idx].cpu().numpy()
                    area = (box[2]-box[0])*(box[3]-box[1])
                    found.append({"box":box, "mask":mask, "area":area, "count":len(res['masks'])})
                    all_areas.append(area)
            detected_objs[obj["id"]] = found


        detailed_object_rewards = {} 
        detailed_spatial_scores = [] 


        for obj in refined_config.get("objects", []):
            obj_id = obj["id"]
            items = detected_objs.get(obj_id, [])
            current_obj_scores = {}
            

            current_obj_scores["object"] = 1.0 if items else 0.0
            

            for key, val in obj.items():
                if key in ["id", "item"]: continue
                
                score_val = 0.0 
                if items:
                    if key in visual_keys:
                        s_list = [calculate_visual_reward(image, i['mask'], i['box'], key, val, obj["item"]) for i in items]
                        score_val = sum(s_list) / len(items)
                    elif key == "number":
                        score_val = calculate_number_reward(items[0]['count'], val)
                    elif key == "size":
                        s_list = [calculate_size_reward(i['area'], all_areas, val) for i in items]
                        score_val = sum(s_list) / len(items)
                
                current_obj_scores[key] = score_val
            
            detailed_object_rewards[obj_id] = current_obj_scores
            

        if "style" in refined_config:
            style_val = refined_config["style"]
            w, h = image.size
            dummy_box = [0, 0, w, h]
            dummy_mask = np.zeros((h, w), dtype=bool) 
            style_score = calculate_visual_reward(image, dummy_mask, dummy_box, "style", style_val, "image")
            detailed_object_rewards["global"] = {"style": style_score}


        if "relation" in refined_config:
            has_depth = any(r["name"] in ["in front of", "behind"] for r in refined_config["relation"])
            depth_map = get_depth_map(image) if has_depth else None

            for rel in refined_config["relation"]:
                id_a = rel.get("ObjectA_id")
                id_b = rel.get("ObjectB_id")
                list_A = _get_objects_safe(detected_objs, id_a)
                list_B = _get_objects_safe(detected_objs, id_b)
                s_score = calculate_relation_reward(list_A, list_B, depth_map, rel["name"])
                detailed_spatial_scores.append(s_score)


        flat_rewards_list = []
        for obj_id, scores in detailed_object_rewards.items():
            for attr_name, score_val in scores.items():
                flat_rewards_list.append(score_val)
        
        flat_rewards_list.extend(detailed_spatial_scores)




        hps_score = 0.0
        prompt_text = prompt
        print("[DEBUG] hps_model is None:", hps_model is None)
        print("[DEBUG] prompt_text:", repr(prompt_text))

        if hps_model is not None and prompt_text:
            with torch.no_grad():

                img_inp = hps_preprocess(image).unsqueeze(0).to(device)
                txt_inp = hps_tokenizer([prompt_text]).to(device)
                

                img_f = hps_model.encode_image(img_inp)
                txt_f = hps_model.encode_text(txt_inp)
                

                img_f /= img_f.norm(dim=-1, keepdim=True)
                txt_f /= txt_f.norm(dim=-1, keepdim=True)
                

                hps_score = (img_f @ txt_f.T).item()
        

        flat_rewards_list.append(hps_score)
        

        if "global" not in detailed_object_rewards:
            detailed_object_rewards["global"] = {}
        detailed_object_rewards["global"]["hpsv2"] = hps_score




        reward_tensor = torch.tensor([flat_rewards_list], dtype=torch.float32, device=device)

        return reward_tensor, detailed_object_rewards, detailed_spatial_scores

    return compute_ours