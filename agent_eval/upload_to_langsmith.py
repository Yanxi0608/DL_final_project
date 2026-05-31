import json
import os
from langsmith import Client

def upload_dataset():
    # 1. 初始化 LangSmith 客户端
    # ⚠️ 提示：在最终的项目交付代码中，建议删掉 api_key 参数，仅依靠环境变量读取，防止密钥泄露。
    client = Client()
    
    dataset_name = "ConferAI-Eval-Set-new"
    
    # 2. 检查数据是否存在
    if not client.has_dataset(dataset_name=dataset_name):
        print(f"正在创建新数据集: {dataset_name}...")
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="学术会议总结智能体多模态评估数据集（含正反均衡样本）"
        )
    else:
        print(f"数据集 {dataset_name} 已存在，正在读取...")
        dataset = client.read_dataset(dataset_name=dataset_name)
        
    # 3. 设置本地文件路径
    json_path = r"D:\DL_final_project\dataset\dataset.json"
    video_dir = r"D:\DL_final_project\dataset\vedios"  # 注意你的文件夹名是 vedios
    
    with open(json_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
        
    print(f"成功读取本地文件，共发现 {len(cases)} 个视频案例。开始上传...")
    
    # 4. 循环遍历并导入到 LangSmith
    for i, case in enumerate(cases, start=1):
        inputs = case.get("inputs", {})
        outputs = case.get("outputs", {})
        metadata = case.get("meta_info", {}) 
        
        # 🌟 关键优化：自动拼接视频的绝对路径，方便下游的评测节点直接调用
        video_filename = inputs.get("video_filename")
        if video_filename:
            # 生成类似 D:/DL_final_project/dataset/vedios/case_01.mp4 的标准路径
            full_video_path = os.path.join(video_dir, video_filename).replace("\\", "/")
            inputs["video_path"] = full_video_path
        
        # 呼叫 LangSmith API 创建样本
        client.create_example(
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            dataset_id=dataset.id
        )
        # 修复了原来打印 inputs.get('video_id') 会输出 None 的问题
        print(f"[{i}/{len(cases)}] 视频 {video_filename} 上传成功！")
        
    print("\n🎉 所有数据已成功同步至 LangSmith！你可以登录 LangSmith 网页端查看 Dataset 页面。")

if __name__ == "__main__":
    upload_dataset()