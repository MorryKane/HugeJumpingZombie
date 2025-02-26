import openai
import time

# 设置你的 OpenAI API key
openai.api_key = "YOUR_API_KEY"

# 1. 上传训练文件，注意：jsonl 文件必须符合 OpenAI 要求的格式
print("上传训练文件中...")
upload_response = openai.File.create(
    file=open("hjz_fine_tuning_dataset.jsonl", "rb"),
    purpose='fine-tune'
)
training_file_id = upload_response.id
print(f"文件上传成功，文件 ID: {training_file_id}")

# 2. 启动 fine tuning 任务，指定 fine tuning 的模型为 "gpt-4o"
print("开始启动 fine tuning 任务...")
fine_tune_job = openai.FineTune.create(
    training_file=training_file_id,
    model="gpt-4o"
)
job_id = fine_tune_job.id
print(f"Fine tuning 任务已启动，任务 ID: {job_id}")

# 3. 轮询任务状态，直到任务完成（succeeded 或 failed）
print("监控任务状态中...")
while True:
    status_response = openai.FineTune.retrieve(id=job_id)
    status = status_response.status
    print(f"当前任务状态: {status}")
    
    if status in ["succeeded", "failed"]:
        break
    time.sleep(30)  # 每 30 秒检查一次状态

print("Fine tuning 任务结束。")
