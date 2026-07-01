from collections import defaultdict
import contextlib
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import flow_grpo.prompts
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel, get_peft_model_state_dict
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper
import torch.nn.functional as F

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class BalancedGRPODataset(Dataset):
    def __init__(self, 
                 config_path_a, sentence_path_a, 
                 config_path_b, sentence_path_b):
        



        def calculate_reward_count(metadata):
            count = 0
            
            for obj in metadata.get('objects', []):

                keys = [k for k in obj.keys() if k != 'id']
                count += len(keys)
                

            count += len(metadata.get('relation', []))

            if 'style' in metadata and metadata['style']:
                count += 1
                
            return count

        def load_data(c_path, s_path):
            if not os.path.exists(c_path) or not os.path.exists(s_path):
                raise FileNotFoundError(f"Files not found: {c_path}, {s_path}")
            
            with open(c_path, 'r', encoding='utf-8') as f:
                c_data = json.load(f)
                c_map = {item['index']: item for item in c_data}
            
            with open(s_path, 'r', encoding='utf-8') as f:
                s_data = json.load(f)
                s_map = {item['index']: item for item in s_data}
            
            merged = []
            seen_prompts = set() 

            local_max_rewards = 0 
            
            common = sorted(list(set(c_map.keys()) & set(s_map.keys())))
            for idx in common:
                prompt_text = s_map[idx]['sentence']
                
                if prompt_text in seen_prompts:
                    continue
                seen_prompts.add(prompt_text)
                
                meta = c_map[idx]
                
                r_count = calculate_reward_count(meta)
                if r_count > local_max_rewards:
                    local_max_rewards = r_count
                
                merged.append({
                    "prompt": prompt_text,
                    "metadata": meta,
                    "reward_count": r_count 
                })
            
            return merged, local_max_rewards

        self.data_a, max_a = load_data(config_path_a, sentence_path_a)
        self.data_b, max_b = load_data(config_path_b, sentence_path_b)
        
        self.global_max_reward_count = max(max_a, max_b)
        
        self.len_a = len(self.data_a)
        self.len_b = len(self.data_b)
        self.total_len = self.len_a + self.len_b
        
        print(f"✅ Dataset Loaded with Pre-calculated Reward Counts.")
        print(f"   - Max Rewards in A: {max_a}")
        print(f"   - Max Rewards in B: {max_b}")
        print(f"   - 🚀 FINAL GLOBAL TARGET PADDING: {self.global_max_reward_count}")

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < self.len_a:
            return self.data_a[idx]
        else:
            return self.data_b[idx - self.len_a]
    
    @staticmethod
    def collate_fn(batch):
        prompts = [item['prompt'] for item in batch]
        metadatas = [item['metadata'] for item in batch]
        
        return {
            "prompts": prompts, 
            "metadata": metadatas
        }
    
class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        """
        Code 1 기반 + A/B 교차 출력 기능 추가
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        

        self.step = 0



        self.len_a = dataset.len_a
        self.len_b = dataset.len_b


        self.total_samples_per_step = self.num_replicas * self.batch_size
        

        assert self.total_samples_per_step % self.k == 0, \
            f"Total batch size ({self.total_samples_per_step}) must be divisible by k ({k})"
        

        self.m = self.total_samples_per_step // self.k

        print(f"k : {k}, num_replicas : {num_replicas}, batch_size : {batch_size}, total_samples : {self.total_samples_per_step}, m : {self.m}")


    def __iter__(self):

        while True:


            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            

            is_turn_a = (self.epoch % 2 == 0)
            
            if is_turn_a:

                current_len = self.len_a
                offset = 0
            else:

                current_len = self.len_b
                offset = self.len_a


            indices = []
            while len(indices) < self.m:

                perm = torch.randperm(current_len, generator=g).tolist()

                perm = [p + offset for p in perm]
                indices.extend(perm)
            

            indices = indices[:self.m]


            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            shuffled_samples = repeated_indices 



            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])


            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


        self.step = 0



def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds

def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    Calculate the proportion of unique prompts whose reward standard deviation is zero.
    
    Args:
        prompts: List of prompts.
        gathered_rewards: Dictionary containing rewards, must include the key 'ori_avg'.
        
    Returns:
        zero_std_ratio: Proportion of prompts with zero standard deviation.
        prompt_std_devs: Mean standard deviation across all unique prompts.
    """

    prompt_array = np.array(prompts)
    

    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    

    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    

    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    

    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    return zero_std_ratio, prompt_std_devs.mean()

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:

        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')
        seed = (base_seed + prompt_hash_int) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

        
def compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config):
    if config.train.cfg:
        noise_pred = transformer(
            hidden_states=torch.cat([sample["latents"][:, j]] * 2),
            timestep=torch.cat([sample["timesteps"][:, j]] * 2),
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = (
            noise_pred_uncond
            + config.sample.guidance_scale
            * (noise_pred_text - noise_pred_uncond)
        )
    else:
        noise_pred = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
    

    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t

def eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)


    all_rewards = defaultdict(list)
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, 
            text_encoders, 
            tokenizers, 
            max_sequence_length=128, 
            device=accelerator.device
        )

        if len(prompt_embeds)<len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]
            sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:len(prompt_embeds)]
        with autocast():
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution, 
                    noise_level=0,
                )
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)

        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)
    
    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
    last_batch_prompt_ids = tokenizers[0](
        prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))

            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            for key, value in all_rewards.items():
                print(key, value.shape)
            accelerator.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
                },
                step=global_step,
            )
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config, pipeline):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        transformer_lora_state_dict = get_peft_model_state_dict(unwrap_model(transformer, accelerator))


        pipeline.save_lora_weights(
            save_directory=save_root_lora,
            transformer_lora_layers=transformer_lora_state_dict,
            safe_serialization=True,
        )
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

def main(_):

    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id


    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
    config.run_name += "_gdpo_hyperconceptmix"
    config.save_dir = os.path.join(config.save_dir, config.run_name)
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,



        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="flow-grpo",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    logger.info(f"\n{config}")


    set_seed(config.seed, device_specific=True)


    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model
    )

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]


    pipeline.safety_checker = None

    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )



    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16


    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=inference_dtype)
    
    pipeline.transformer.to(accelerator.device)

    if config.use_lora:

        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)

            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
    
    transformer = pipeline.transformer
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)
    


    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True


    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )


    reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)

    if config.prompt_fn == "ours":
        train_dataset = BalancedGRPODataset(
            config_path_a="./dataset/hyperconcepmix_train/3000_refined_config_k=7.json", 
            sentence_path_a="./dataset/hyperconcepmix_train/3000_sentences_k=7.json",
            config_path_b="./dataset/hyperconcepmix_train/final_config_all.json", 
            sentence_path_b="./dataset/hyperconcepmix_train/final_sentences_all.json"
        )

        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42,
        )


        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0,
            collate_fn=BalancedGRPODataset.collate_fn,

        )
   
    else:
        raise NotImplementedError("Only general_ocr is supported with dataset")

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False

    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)



    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast



    transformer, optimizer, train_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader)



    executor = futures.ThreadPoolExecutor(max_workers=8)


    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )

    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")




    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    while True:

        pipeline.transformer.eval()


        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config, pipeline)


        pipeline.transformer.eval()
        samples = []
        prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            if config.prompt_fn == "ours":
                batch = next(train_iter) 
                prompts = batch["prompts"]          
                prompt_metadata = batch["metadata"]
            else:
                prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, 
                text_encoders, 
                tokenizers, 
                max_sequence_length=128, 
                device=accelerator.device
            )
            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)


            if config.sample.same_latent:
                generator = create_generator(prompts, base_seed=epoch*10000+i)
            else:
                generator = None
            with autocast():
                with torch.no_grad():
                    images, latents, log_probs = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution, 
                        noise_level=config.sample.noise_level,
                        generator=generator
                )

            latents = torch.stack(
                latents, dim=1
            )
            log_probs = torch.stack(log_probs, dim=1)

            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.train_batch_size, 1
            )


            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)

            time.sleep(0)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[
                        :, :-1
                    ],
                    "next_latents": latents[
                        :, 1:
                    ],
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )


        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()

            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float().squeeze(1)
                for key, value in rewards.items()
            }

        max_reward_len = train_dataset.global_max_reward_count
        for s in samples:
            if 'rewards' in s and 'scores' in s['rewards']:
                tensor = s['rewards']['scores']
                


                current_len = tensor.shape[-1] 
                
                if current_len < max_reward_len:
                    diff = max_reward_len - current_len

                    s['rewards']['scores'] = F.pad(tensor, (0, diff), value=-1.0)
                    

                    if 'ori_scores' in s['rewards']:
                         s['rewards']['ori_scores'] = F.pad(s['rewards']['ori_scores'], (0, diff), value=-1.0)


        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        if epoch % 5 == 0 and accelerator.is_main_process:

            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                sampled_prompts = [prompts[i] for i in sample_indices]

                accelerator.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt:.100}",
                            )
                            for idx, prompt in enumerate(sampled_prompts)
                        ],
                    },
                    step=global_step,
                )
        samples["rewards"]["ori_scores"] = samples["rewards"]["scores"]

        samples["rewards"]["scores"] = samples["rewards"]["scores"].unsqueeze(-1).repeat(1,1 , num_train_timesteps)

        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}

        has_padding = (gathered_rewards['ori_scores'] == -1).any(axis=1)
        single_part =gathered_rewards['ori_scores'][has_padding]
        concept_part = gathered_rewards['ori_scores'][~has_padding]
        concept_mean = concept_part[concept_part != -1].mean()
        single_mean = single_part[single_part != -1].mean()
        valid_rewards = gathered_rewards['ori_scores'][gathered_rewards['ori_scores'] != -1]
 
        accelerator.log(
            {
                "reward": gathered_rewards['ori_scores'],
                "epoch": epoch,
                "reward_mean": valid_rewards.mean(),
                "reward_std": valid_rewards.std(),
                "conceptmix_reward_mean": concept_mean.mean(),
                "single_attr_binding_reward_mean": single_mean.mean(),
            },
            step=global_step,
        )

        if config.per_prompt_stat_tracking:

            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            dummy_rewards = np.zeros(len(prompts))
            advantages = stat_tracker.update(prompts, dummy_rewards, reward_tensors=gathered_rewards['scores'])
            if accelerator.is_local_main_process:
                print("len(prompts)", len(prompts))
                print("len unique prompts", len(set(prompts)))

            group_size, trained_prompt_num = stat_tracker.get_stats()



            if accelerator.is_main_process:
                accelerator.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,


                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards['scores'] - gathered_rewards['scores'].mean()) / (gathered_rewards['scores'].std() + 1e-4)


        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())

        del samples["rewards"]
        del samples["prompt_ids"]


        mask = (samples["advantages"].abs().sum(dim=1) != 0)
        


        num_batches = config.sample.num_batches_per_epoch
        true_count = mask.sum()
        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        if accelerator.is_main_process:
            accelerator.log(
                {
                    "actual_batch_size": mask.sum().item()//config.sample.num_batches_per_epoch,
                },
                step=global_step,
            )

        samples = {k: v[mask] for k, v in samples.items()}

        total_batch_size, num_timesteps = samples["timesteps"].shape




        assert num_timesteps == config.sample.num_steps


        for inner_epoch in range(config.train.num_inner_epochs):

            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}


            samples_batched = {
                k: v.reshape(-1, total_batch_size//config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }


            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]


            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg:

                    embeds = torch.cat(
                        [train_neg_prompt_embeds[:len(sample["prompt_embeds"])], sample["prompt_embeds"]]
                    )
                    pooled_embeds = torch.cat(
                        [train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])], sample["pooled_prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]
                    pooled_embeds = sample["pooled_prompt_embeds"]

                train_timesteps = [step_index  for step_index in range(num_train_timesteps)]
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config)
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    with transformer.module.disable_adapter():
                                        _, _, prev_sample_mean_ref, _ = compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config)


                        advantages = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        if config.train.beta > 0:
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * std_dev_t ** 2)
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_gt_one"].append(
                            torch.mean(
                                (
                                    ratio - 1.0 > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_lt_one"].append(
                            torch.mean(
                                (
                                    1.0 - ratio > config.train.clip_range
                                ).float()
                            )
                        )
                        info["policy_loss"].append(policy_loss)
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss)

                        info["loss"].append(loss)


                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()


                    if accelerator.sync_gradients:

                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)


        
        epoch+=1
        
if __name__ == "__main__":
    app.run(main)

