import gradio as gr
import requests
import json
import os
from PIL import Image
import tempfile
import time

def generate_motion(gpu_id, output_ext, text_input, text_file):
    api_url = "http://localhost:8000/generate_motion"
    
    # 准备请求数据
    request_data = {
        "gpu_id": int(gpu_id),
        "ext": output_ext,
        "output_dir": os.path.abspath("generation")  # 添加输出基础目录参数
    }
    
    # 判断输入方式
    if text_input:
        request_data["text_prompt"] = text_input
    elif text_file is not None:
        # 读取文件内容而不是发送文件路径
        with open(text_file.name, 'r') as f:
            text_content = f.read().strip()
        request_data["text_prompt"] = text_content
    else:
        return "错误：请输入文字描述或上传文本文件", None

    try:
        # 确保generation目录存在
        base_dir = os.path.abspath("generation")
        output_dir = os.path.join(base_dir, output_ext)
        animations_dir = os.path.join(output_dir, "animations", "0")
        os.makedirs(animations_dir, exist_ok=True)
        
        # 发送请求到API
        response = requests.post(api_url, json=request_data)
        response.raise_for_status()
        result = response.json()
        
        if result["status"] != "success":
            return f"生成失败：{result.get('detail', '未知错误')}", None
        
        # 处理结果
        output_text = [
            f"状态：{result['status']}",
            f"输出目录：{result['output_dir']}",
            f"消息：{result['message']}"
        ]
        
        # 等待一段时间确保文件已生成
        time.sleep(2)
        
        try:
            # 查找生成的视频文件
            # 优先查找带IK的视频文件
            video_files = [f for f in os.listdir(animations_dir) if f.endswith('_ik.mp4')]
            if not video_files:
                # 如果没有找到带IK的视频，则查找普通视频
                video_files = [f for f in os.listdir(animations_dir) if f.endswith('.mp4')]
                
            if video_files:
                video_path = os.path.join(animations_dir, video_files[0])
                if os.path.exists(video_path):
                    return "\n".join(output_text), video_path
                
            return "\n".join(output_text) + "\n警告：未找到生成的视频文件", None
            
        except FileNotFoundError:
            return "\n".join(output_text) + f"\n错误：无法访问目录 {animations_dir}", None
    
    except requests.exceptions.RequestException as e:
        return f"API请求错误：{str(e)}", None
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"发生错误：{str(e)}\n{error_details}", None

# 创建Gradio界面
with gr.Blocks(title="文本生成动作系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 文本生成动作系统")
    
    with gr.Row():
        with gr.Column(scale=1):
            gpu_id = gr.Number(
                label="GPU ID",
                value=0,
                minimum=-1,
                maximum=8,
                step=1,
                interactive=True
            )
            output_ext = gr.Textbox(
                label="输出文件夹名称",
                value="default",
                interactive=True
            )
            
        
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="输入动作描述文本",
                placeholder="请输入描述动作的英文文本...",
                lines=3
            )
            text_file = gr.File(
                label="或上传文本文件（每行一个描述）",
                file_types=[".txt"]
            )
    
    generate_btn = gr.Button("生成动作", variant="primary")
    
    with gr.Column():
        output_text = gr.Markdown(label="生成结果")
        output_video = gr.Video(label="生成的动画",height=480,width=300)
    
    # 设置点击事件
    generate_btn.click(
        fn=generate_motion,
        inputs=[gpu_id, output_ext, text_input, text_file],
        outputs=[output_text, output_video]
    )

# 启动Gradio应用
if __name__ == "__main__":
    # 确保基础目录存在
    base_dir = os.path.abspath("generation")
    os.makedirs(base_dir, exist_ok=True)
    
    # 启动应用
    demo.launch(share=True, server_name="0.0.0.0")