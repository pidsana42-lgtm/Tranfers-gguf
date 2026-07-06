# คู่มือการแปลงโมเดล Phonsiri/Gemma-4-E4B-it-PARL เป็น GGUF 4-bit บน Cloud (Google Colab / RunPod)

คู่มือนี้แนะนำขั้นตอนและชุดโค้ดสำหรับดาวน์โหลดโมเดล **Phonsiri/Gemma-4-E4B-it-PARL** จาก Hugging Face เพื่อแปลงเป็นฟอร์แมต **GGUF** และทำ **4-bit Quantization** (แนะนำแบบ `Q4_K_M`) บนระบบ Cloud เช่น Google Colab หรือ Cloud VM อื่นๆ

---

## 🛠️ แนะนำการเตรียมตัวก่อนเริ่ม (บน Google Colab)
1. เปิด **Google Colab** (https://colab.research.google.com/)
2. เลือก Runtime เป็น **T4 GPU** หรือ GPU อื่นๆ (ไปที่ *Runtime > Change runtime type > T4 GPU*) เพื่อช่วยให้การดาวน์โหลดและประมวลผลทำได้เร็วขึ้น
3. นำไฟล์สคริปต์ [convert_to_gguf.sh](file:///Users/phonsirithabunsri/Desktop/Tranfer/convert_to_gguf.sh) หรือ [convert_to_gguf.py](file:///Users/phonsirithabunsri/Desktop/Tranfer/convert_to_gguf.py) ไปอัปโหลดขึ้น Colab หรือแก้ไขค่าในสคริปต์ก่อนรัน

---

## 🔑 วิธีการกรอกข้อมูล Hugging Face เพื่ออัปโหลดอัตโนมัติ
ในหัวไฟล์ของทั้งสคริปต์ Bash และ Python จะมีส่วนให้กำหนดตัวแปรไว้ ให้กรอกค่า Username และ Write Token ของคุณลงไปก่อนเริ่มรัน:

### ในไฟล์ Bash (`convert_to_gguf.sh`):
```bash
# Hugging Face Configuration (Fill these in to automatically upload the converted model)
HF_USERNAME="ชื่อยูสเซอร์เนมของคุณ" 
HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxx" 
```

### ในไฟล์ Python (`convert_to_gguf.py`):
```python
# Hugging Face Configuration (Fill these in to automatically upload the converted model)
HF_USERNAME = "ชื่อยูสเซอร์เนมของคุณ"
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxx"
```

*หมายเหตุ: หากเว้นช่องเหล่านี้ว่างไว้ สคริปต์จะแปลงโมเดลเสร็จสมบูรณ์ตามปกติ แต่จะข้ามขั้นตอนอัปโหลดขึ้น Hugging Face*

---

## 📋 ขั้นตอนการรันโค้ดบน Colab ทีละสเต็ป (แบบแมนนวล)

### Cell 1: ติดตั้งไลบรารีที่จำเป็นและดึง `llama.cpp`
```bash
# 1. ติดตั้ง System dependencies และคอมไพล์ llama.cpp
!git clone https://github.com/ggerganov/llama.cpp.git
%cd llama.cpp
!make -j$(nproc)

# 2. ติดตั้ง Python requirements สำหรับแปลงโมเดล
!pip install -r requirements.txt
!pip install huggingface_hub torch transformers accelerator sentencepiece
```

### Cell 2: ดาวน์โหลดโมเดลต้นฉบับ
```python
import os
from huggingface_hub import snapshot_download

model_id = "Phonsiri/Gemma-4-E4B-it-PARL"
local_dir = "/content/Gemma-4-E4B-it-PARL"

print(f"กำลังดาวน์โหลดโมเดล {model_id}...")
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    ignore_patterns=["*.gguf", "*.bin"]
)
print("ดาวน์โหลดสำเร็จแล้ว!")
```

### Cell 3: แปลงโมเดลเป็น GGUF (FP16)
```bash
!python3 convert_hf_to_gguf.py /content/Gemma-4-E4B-it-PARL \
    --outfile /content/Gemma-4-E4B-it-PARL-f16.gguf \
    --outtype f16
```

### Cell 4: ทำ Quantization เป็น 4-bit (Q4_K_M)
```bash
!./llama-quantize /content/Gemma-4-E4B-it-PARL-f16.gguf /content/Gemma-4-E4B-it-PARL-Q4_K_M.gguf Q4_K_M
```

### Cell 5: อัปโหลดโมเดลขึ้น Hugging Face (กรอก API Token)
```python
from huggingface_hub import HfApi

hf_username = ""         # กรอกชื่อผู้ใช้ Hugging Face ของคุณที่นี่
hf_token = ""            # กรอก Write Token ของคุณที่นี่ (เว้นไว้หากต้องการอัปโหลดเองภายหลัง)
new_repo_name = "Gemma-4-E4B-it-PARL-GGUF"

if hf_username and hf_token:
    api = HfApi()
    repo_id = f"{hf_username}/{new_repo_name}"
    
    # สร้าง repository ใหม่
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=hf_token)
    
    # อัปโหลดไฟล์
    print("กำลังอัปโหลดโมเดล GGUF ไปยัง Hugging Face...")
    api.upload_file(
        path_or_fileobj="/content/Gemma-4-E4B-it-PARL-Q4_K_M.gguf",
        path_in_repo="Gemma-4-E4B-it-PARL-Q4_K_M.gguf",
        repo_id=repo_id,
        repo_type="model",
        token=hf_token
    )
    print("อัปโหลดเรียบร้อยแล้ว!")
else:
    print("ไม่ได้กำหนดข้อมูลผู้ใช้หรือ Token ไว้ ข้ามขั้นตอนการอัปโหลด")
```
