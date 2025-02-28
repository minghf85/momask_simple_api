from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import os
from os.path import join as pjoin
import numpy as np

# 导入gen_t2m.py中的必要函数和类
from gen_t2m import (
    load_vq_model, 
    load_trans_model,
    load_res_model,
    load_len_estimator,
    EvalT2MOptions,
    get_opt,
    fixseed
)
from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion
from visualization.joints2bvh import Joint2BVHConvertor
from utils.paramUtil import t2m_kinematic_chain

app = FastAPI()

# 请求模型
class MotionRequest(BaseModel):
    gpu_id: int = 0
    ext: str
    output_dir: str  # 添加输出基础目录参数
    text_prompt: Optional[str] = None
    text_path: Optional[str] = None

# 响应模型
class MotionResponse(BaseModel):
    status: str
    output_dir: str
    message: str

# 全局变量存储加载的模型
loaded_models = {}

def load_models(gpu_id: int):
    """加载所有必要的模型"""
    if gpu_id in loaded_models:
        return loaded_models[gpu_id]
    
    parser = EvalT2MOptions()
    opt = parser.parse()
    opt.gpu_id = gpu_id
    fixseed(opt.seed)
    
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else f"cuda:{str(opt.gpu_id)}")
    torch.autograd.set_detect_anomaly(True)
    
    dim_pose = 251 if opt.dataset_name == 'kit' else 263
    
    # 加载VQ模型
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    print(f"Reading {vq_opt_path}")
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_opt.dim_pose = dim_pose
    vq_model, vq_opt = load_vq_model(vq_opt)
    
    # 加载其他模型
    model_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'opt.txt')
    print(f"Reading {model_opt_path}")
    model_opt = get_opt(model_opt_path, device=opt.device)
    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim
    
    # 加载R-Transformer
    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
    print(f"Reading {res_opt_path}")
    res_opt = get_opt(res_opt_path, device=opt.device)
    print(f"latent_dim: {res_opt.latent_dim}, ff_size: {res_opt.ff_size}, nlayers: {res_opt.n_layers}, nheads: {res_opt.n_heads}, dropout: {res_opt.dropout}")
    print("Loading CLIP...")
    res_model = load_res_model(res_opt, vq_opt, opt)
    
    # 加载M-Transformer
    print(f"latent_dim: {model_opt.latent_dim}, ff_size: {model_opt.ff_size}, nlayers: {model_opt.n_layers}, nheads: {model_opt.n_heads}, dropout: {model_opt.dropout}")
    print("Loading CLIP...")
    t2m_transformer = load_trans_model(model_opt, opt, 'latest.tar')
    
    # 加载Length Predictor
    length_estimator = load_len_estimator(model_opt)
    
    # 将模型移至指定设备
    vq_model.to(opt.device)
    res_model.to(opt.device)
    t2m_transformer.to(opt.device)
    length_estimator.to(opt.device)
    
    # 设置为评估模式
    vq_model.eval()
    res_model.eval()
    t2m_transformer.eval()
    length_estimator.eval()
    
    # 加载均值和标准差
    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))
    
    loaded_models[gpu_id] = {
        'vq_model': vq_model,
        't2m_transformer': t2m_transformer,
        'res_model': res_model,
        'length_estimator': length_estimator,
        'opt': opt,
        'mean': mean,
        'std': std
    }
    
    return loaded_models[gpu_id]

@app.post("/generate_motion", response_model=MotionResponse)
async def generate_motion(request: MotionRequest):
    try:
        # 验证输入
        if not request.text_prompt and not request.text_path:
            raise HTTPException(status_code=400, detail="必须提供text_prompt或text_path其中之一")
            
        # 使用请求中指定的输出目录
        result_dir = os.path.join(request.output_dir, request.ext)
        joints_dir = os.path.join(result_dir, 'joints')
        animation_dir = os.path.join(result_dir, 'animations')
        
        # 确保输出目录存在
        os.makedirs(joints_dir, exist_ok=True)
        os.makedirs(animation_dir, exist_ok=True)
        
        # 加载模型
        models = load_models(request.gpu_id)
        opt = models['opt']
        
        # 准备文本提示列表
        prompt_list = []
        length_list = []
        est_length = False
        
        if request.text_prompt:
            prompt_list.append(request.text_prompt)
            est_length = True
            print("Since no motion length are specified, we will use estimated motion lengthes!!")
        elif request.text_path:
            if not os.path.exists(request.text_path):
                raise HTTPException(status_code=400, detail="text_path文件不存在")
            with open(request.text_path, 'r') as f:
                for line in f:
                    infos = line.split('#')
                    prompt_list.append(infos[0])
                    if len(infos) == 1 or (not infos[1].isdigit()):
                        est_length = True
                        length_list = []
                        print("Since no motion length are specified, we will use estimated motion lengthes!!")
                    else:
                        length_list.append(int(infos[-1]))
        
        print("-->Repeat 0")
        # 生成动作
        with torch.no_grad():
            if est_length:
                # 预测长度
                text_embedding = models['t2m_transformer'].encode_text(prompt_list)
                pred_dis = models['length_estimator'](text_embedding)
                probs = F.softmax(pred_dis, dim=-1)
                token_lens = Categorical(probs).sample()
            else:
                token_lens = torch.LongTensor(length_list) // 4
                token_lens = token_lens.to(opt.device).long()
            
            m_length = token_lens * 4
            
            for k, (caption, length) in enumerate(zip(prompt_list, m_length)):
                print(f"---->Sample {k}: {caption} {length.item()}")
            
            # 生成动作序列
            mids = models['t2m_transformer'].generate(
                prompt_list, 
                token_lens,
                timesteps=opt.time_steps,
                cond_scale=opt.cond_scale,
                temperature=opt.temperature,
                topk_filter_thres=opt.topkr,
                gsample=opt.gumbel_sample
            )
            
            mids = models['res_model'].generate(mids, prompt_list, token_lens, temperature=1, cond_scale=5)
            pred_motions = models['vq_model'].forward_decoder(mids)
            pred_motions = pred_motions.detach().cpu().numpy()
            
            # 反归一化
            data = pred_motions * models['std'] + models['mean']
            
            # 处理每个生成的动作
            converter = Joint2BVHConvertor()
            for k, (caption, joint_data) in enumerate(zip(prompt_list, data)):
                print(f"---->Sample {k}: {caption} {m_length[k].item()}")
                animation_path = pjoin(animation_dir, str(k))
                joint_path = pjoin(joints_dir, str(k))
                os.makedirs(animation_path, exist_ok=True)
                os.makedirs(joint_path, exist_ok=True)
                
                joint_data = joint_data[:m_length[k]]
                joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()
                
                # 保存带IK的BVH
                bvh_path = pjoin(animation_path, f"sample{k}_len{m_length[k]}_ik.bvh")
                _, ik_joint = converter.convert(joint, filename=bvh_path, iterations=100)
                
                # 保存不带IK的BVH
                bvh_path = pjoin(animation_path, f"sample{k}_len{m_length[k]}.bvh")
                _, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)
                
                # 保存动画和关节数据
                save_path = pjoin(animation_path, f"sample{k}_len{m_length[k]}.mp4")
                ik_save_path = pjoin(animation_path, f"sample{k}_len{m_length[k]}_ik.mp4")
                
                plot_3d_motion(ik_save_path, t2m_kinematic_chain, ik_joint, title=caption, fps=20)
                plot_3d_motion(save_path, t2m_kinematic_chain, joint, title=caption, fps=20)
                
                np.save(pjoin(joint_path, f"sample{k}_len{m_length[k]}.npy"), joint)
                np.save(pjoin(joint_path, f"sample{k}_len{m_length[k]}_ik.npy"), ik_joint)
            
        return MotionResponse(
            status="success",
            output_dir=result_dir,
            message=f"成功生成{len(prompt_list)}个动作序列"
        )
        
    except Exception as e:
        import traceback
        error_msg = f"错误详情：{str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)