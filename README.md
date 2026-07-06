# คู่มือการแปลงโมเดล Phonsiri/Gemma-4-E4B-it-PARL เป็น GGUF 4-bit บน Cloud (Google Colab / RunPod / Lightning AI)

คู่มือนี้แนะนำขั้นตอนและชุดโค้ดสำหรับดาวน์โหลดโมเดล **Phonsiri/Gemma-4-E4B-it-PARL** จาก Hugging Face เพื่อแปลงเป็นฟอร์แมต **GGUF** และทำ **4-bit Quantization** (แนะนำแบบ `Q4_K_M`) บนระบบ Cloud เช่น Google Colab หรือ Cloud VM อื่นๆ

---

## 🛠️ แนะนำการเตรียมตัวก่อนเริ่ม (บน Google Colab / Cloud Studio)
1. เปิด **Google Colab** (https://colab.research.google.com/) หรือ Cloud Space ของคุณ
2. เลือก Runtime เป็น **T4 GPU** หรือ GPU อื่นๆ
3. นำไฟล์สคริปต์ [convert_to_gguf.sh](file:///Users/phonsirithabunsri/Desktop/Tranfer/convert_to_gguf.sh) หรือ [convert_to_gguf.py](file:///Users/phonsirithabunsri/Desktop/Tranfer/convert_to_gguf.py) ไปอัปโหลดขึ้น Colab หรือแก้ไขค่าในสคริปต์ก่อนรัน

---

## 🔑 วิธีการกรอกข้อมูล Hugging Face เพื่ออัปโหลดอัตโนมัติ
ในหัวไฟล์ของทั้งสคริปต์ Bash และ Python จะมีส่วนให้กำหนดตัวแปรไว้ ให้กรอกค่า Username และ Write Token ของคุณลงไปก่อนเริ่มรัน:

### ในไฟล์ Bash (`convert_to_gguf.sh`):
```bash
# Hugging Face Configuration (Fill these in to automatically upload the converted model)
HF_USERNAME="Phonsiri" 
HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxx" 
HF_REPO_NAME="Gemma-4-E4B-it-PARL-GGUF"
```

### ในไฟล์ Python (`convert_to_gguf.py`):
```python
# Hugging Face Configuration (Fill these in to automatically upload the converted model)
HF_USERNAME = "Phonsiri"
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxx"
HF_REPO_NAME = "Gemma-4-E4B-it-PARL-GGUF"
```

*หมายเหตุ: หากเว้นช่องเหล่านี้ว่างไว้ สคริปต์จะแปลงโมเดลเสร็จสมบูรณ์ตามปกติ แต่จะข้ามขั้นตอนอัปโหลดขึ้น Hugging Face*

---

## 🚀 วิธีที่ง่ายที่สุด: รันผ่านสคริปต์อัตโนมัติ (รันคำสั่งเดียวเสร็จหมด)

หากคุณอยู่ที่โฟลเดอร์รากของโปรเจกต์ (`~/Tranfers-gguf`) คุณสามารถรันสคริปต์ Python หรือ Bash ที่เราสร้างให้เพื่อดำเนินการทั้งหมดโดยอัตโนมัติ (ติดตั้ง dependencies, คอมไพล์, โหลดโมเดล, แปลงไฟล์, quantize, และอัปโหลด):

```bash
# วิธีที่ 1: รันด้วย Python
python3 convert_to_gguf.py

# วิธีที่ 2: รันด้วย Bash (กำหนดสิทธิ์ในการรันก่อน)
chmod +x convert_to_gguf.sh
./convert_to_gguf.sh
```

---

## 📋 ขั้นตอนการรันโค้ดแบบแมนนวล (รันทีละคำสั่งในโฟลเดอร์โปรเจกต์)

หากต้องการทำทีละขั้นตอนด้วยตนเองจากไดเรกทอรีหลัก (`~/Tranfers-gguf`) ให้รันดังนี้:

### ขั้นที่ 1: ติดตั้งไลบรารีที่จำเป็นและคอมไพล์ `llama.cpp`
```bash
# 1. โคลนและคอมไพล์ llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make -j$(nproc)
pip install -r requirements.txt
cd ..

# 2. ติดตั้งแพ็กเกจเสริม
pip install huggingface_hub torch transformers accelerator sentencepiece
```

### ขั้นที่ 2: ดาวน์โหลดโมเดลต้นฉบับจาก Hugging Face
```python
import os
from huggingface_hub import snapshot_download

model_id = "Phonsiri/Gemma-4-E4B-it-PARL"
local_dir = "./Gemma-4-E4B-it-PARL"

print(f"กำลังดาวน์โหลดโมเดล {model_id}...")
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    ignore_patterns=["*.gguf", "*.bin"]
)
print("ดาวน์โหลดสำเร็จแล้ว!")
```

### ขั้นที่ 3: แปลงโมเดลเป็น GGUF (FP16)
เนื่องจากไฟล์สคริปต์สำหรับแปลง อยู่ในโฟลเดอร์ `llama.cpp` หากคุณอยู่ที่โฟลเดอร์หลัก (`~/Tranfers-gguf`) ต้องอ้างอิงตำแหน่งไฟล์ด้วย `llama.cpp/convert_hf_to_gguf.py`:

```bash
python3 llama.cpp/convert_hf_to_gguf.py ./Gemma-4-E4B-it-PARL \
    --outfile ./Gemma-4-E4B-it-PARL-f16.gguf \
    --outtype f16
```

### ขั้นที่ 4: ทำ Quantization เป็น 4-bit (Q4_K_M)
เช่นเดียวกันกับเครื่องมือ quantize ต้องอ้างอิงตำแหน่งไฟล์ด้วย `llama.cpp/llama-quantize`:

```bash
./llama.cpp/llama-quantize ./Gemma-4-E4B-it-PARL-f16.gguf ./Gemma-4-E4B-it-PARL-Q4_K_M.gguf Q4_K_M
```

### ขั้นที่ 5: อัปโหลดโมเดลขึ้น Hugging Face (กรอก API Token)
```python
from huggingface_hub import HfApi

hf_username = "Phonsiri"                  # กรอกชื่อผู้ใช้ Hugging Face ของคุณที่นี่
hf_token = "hf_xxxxxxxxxxxxxxxx"          # กรอก Write Token ของคุณที่นี่
hf_repo_name = "Gemma-4-E4B-it-PARL-GGUF"

if hf_username and hf_token:
    api = HfApi()
    repo_id = f"{hf_username}/{hf_repo_name}"
    
    # สร้าง repository ใหม่
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=hf_token)
    
    # อัปโหลดไฟล์
    print("กำลังอัปโหลดโมเดล GGUF ไปยัง Hugging Face...")
    api.upload_file(
        path_or_fileobj="./Gemma-4-E4B-it-PARL-Q4_K_M.gguf",
        path_in_repo="Gemma-4-E4B-it-PARL-Q4_K_M.gguf",
        repo_id=repo_id,
        repo_type="model",
        token=hf_token
    )
    print("อัปโหลดเรียบร้อยแล้ว!")
else:
    print("ไม่ได้กำหนดข้อมูลผู้ใช้หรือ Token ไว้ ข้ามขั้นตอนการอัปโหลด")
```
