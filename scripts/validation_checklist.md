# Testing & Validation Guide สำหรับ YOLO v8 Improvements

## 📋 ภาพรวม

เอกสารนี้อธิบายวิธีการทดสอบและวัดผลการปรับปรุง YOLO v8 อย่างเป็นระบบ

---

## 1. การเตรียมการ

### 1.1 ติดตั้ง Dependencies
```bash
pip install ultralytics torch torchvision pandas matplotlib seaborn
pip install pytest pytest-cov  # สำหรับ testing
```

### 1.2 เตรียม Dataset
```bash
# ดาวน์โหลด dataset สำหรับ testing
# Coco128 เป็น dataset เล็กสำหรับ quick test
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').val(data='coco128.yaml')"
```

---

## 2. การสร้าง Baseline Metrics

### 2.1 รัน Baseline Benchmark
```bash
# สร้าง baseline metrics ก่อนปรับปรุง
python scripts/run_benchmarks.py --mode baseline --model yolov8n.pt --data coco128.yaml

# หรือใช้ GPU
python scripts/run_benchmarks.py --mode baseline --model yolov8n.pt --data coco128.yaml --device 0
```

ผลลัพธ์จะถูกบันทึกที่:
- `runs/baseline_metrics.json` - Baseline metrics
- `runs/baseline/benchmark_summary.txt` - สรุปผลแบบอ่านง่าย

### 2.2 Metrics ที่ต้องบันทึก
- **Accuracy**: mAP50, mAP50-95, Precision, Recall, Fitness
- **Speed**: FPS, Latency (ms)
- **Memory**: Peak Memory (MB), GPU Memory Usage
- **Per-class**: mAP ต่อ class (สำหรับ class imbalance analysis)

---

## 3. การทดสอบหลังปรับปรุง

### 3.1 หลังจากปรับปรุงโค้ดแล้ว

```bash
# รัน validation และเปรียบเทียบกับ baseline
python scripts/run_benchmarks.py --mode compare --model yolov8n.pt --data coco128.yaml
```

### 3.2 ใช้ Python API

```python
from tests.test_benchmark import BenchmarkRunner, AblationStudy

# รัน quick benchmark
runner = BenchmarkRunner(
    model_path='yolov8n.pt',
    data_config='coco128.yaml',
    device='cuda:0'  # หรือ 'cpu'
)

results = runner.run_full_validation(batch_size=16, imgsz=640)
print(f"mAP50-95: {results['metrics']['mAP50-95']:.4f}")
print(f"FPS: {results['speed']['fps']:.1f}")
```

---

## 4. Ablation Studies (การทดสอบทีละส่วน)

### 4.1 สร้าง Ablation Study

```python
from tests.test_benchmark import AblationStudy

study = AblationStudy(
    baseline_model='yolov8n.pt',
    data_config='coco128.yaml',
    save_dir='runs/ablation'
)

# ทดสอบทีละ improvement
study.add_experiment(
    name='baseline',
    description='Original YOLOv8n'
)

study.add_experiment(
    name='with_focal_loss',
    description='Focal Loss added',
    modifications={'loss': 'focal'}
)

study.add_experiment(
    name='with_gridmask',
    description='GridMask augmentation',
    modifications={'augmentation': 'gridmask'}
)

study.add_experiment(
    name='with_all',
    description='All improvements combined',
    modifications={'loss': 'focal', 'augmentation': 'gridmask'}
)

# รันทุก experiment
df_results = study.run_all()
print(df_results)

# เปรียบเทียบกับ baseline เป็น %
improvements = study.compare_with_baseline()
print(improvements)
```

### 4.2 รันผ่าน Script

```bash
# สร้างไฟล์ Python สำหรับ ablation study
python -c "
from tests.test_benchmark import AblationStudy

study = AblationStudy('yolov8n.pt', 'coco128.yaml')
study.add_experiment('baseline', 'Original model')
study.add_experiment('focal_loss', 'With Focal Loss')
df = study.run_all(device='cpu', batch_size=16, imgsz=640)
"
```

---

## 5. การทดสอบทีละ Improvement

### 5.1 Data Augmentation Testing
```python
# ทดสอบ augmentation ใหม่
python scripts/run_benchmarks.py --mode compare --data coco128.yaml

# เปรียบเทียบ before/after
# วิเคราะห์ผลลัพธ์ที่ runs/baseline/ และ runs/improvements/
```

### 5.2 Loss Function Testing
```python
# หลังจากเปลี่ยน loss function
runner = BenchmarkRunner('yolov8n.pt', 'coco128.yaml')
results = runner.run_full_validation()

# ดูผลกระทบต่อ small objects
# ดูผลกระทบต่อ class imbalance
```

### 5.3 Model Architecture Testing
```python
# ทดสอบ modules ใหม่ (เช่น CBAM, Soft-NMS)
# ต้องสร้าง model weights ใหม่และทดสอบ
python scripts/run_benchmarks.py --mode compare --model path/to/new_model.pt
```

---

## 6. Metrics ที่ควรดู

### 6.1 Primary Metrics
| Metric | ความหมาย | Target |
|--------|----------|--------|
| **mAP50** | Mean Average Precision @ IoU=0.5 | ↑ สูงขึ้น |
| **mAP50-95** | Mean Average Precision @ IoU=0.5-0.95 | ↑ สูงขึ้น |
| **Precision** | ความแม่นยำในการตรวจจับ | ↑ สูงขึ้น |
| **Recall** | ความครอบคลุมในการตรวจจับ | ↑ สูงขึ้น |
| **FPS** | Frames per Second | ↑ เร็วขึ้น |
| **Latency** | เวลาที่ใช้ต่อภาพ (ms) | ↓ ต่ำลง |

### 6.2 Trade-offs ที่ต้องพิจารณา
- **Accuracy vs Speed**: บางทีเพิ่ม accuracy แล้วช้าลง
- **Memory vs Batch Size**: โมเดลใหญ่ใช้ memory เยอะ
- **Precision vs Recall**: ต้อง balance ให้เหมาะสม

---

## 7. Best Practices

### 7.1 ก่อนทำการเปลี่ยนแปลง
```bash
# ✅ สร้าง baseline ก่อน
python scripts/run_benchmarks.py --mode baseline

# ✅ Backup โค้ดเดิม
git checkout -b backup-before-improvements
git commit -m "Backup before improvements"
```

### 7.2 หลังทำการเปลี่ยนแปลง
```bash
# ✅ ทดสอบทีละส่วน
# อย่าเปลี่ยนหลายอย่างพร้อมกัน

# ✅ เปรียบเทียบผลลัพธ์
python scripts/run_benchmarks.py --mode compare

# ✅ บันทึกผลลัพธ์
git add runs/benchmark_results.json
git commit -m "Add benchmark results for [improvement name]"
```

### 7.3 ทำ Ablation Study
```python
# ✅ ทดสอบทีละ improvement
# ดูว่าแต่ละอย่างช่วยอะไรบ้าง

study = AblationStudy(...)
study.add_experiment('1_augmentation', 'GridMask only')
study.add_experiment('2_loss', 'Focal Loss only')
study.add_experiment('3_combined', 'GridMask + Focal Loss')
study.run_all()
```

---

## 8. การวิเคราะห์ผลลัพธ์

### 8.1 ดูไฟล์ผลลัพธ์
```bash
# ดู summary
cat runs/baseline/benchmark_summary.txt

# ดู JSON ละเอียด
cat runs/baseline/benchmark_results.json | jq .

# เปรียบเทียบ
diff runs/baseline/benchmark_results.json runs/current/benchmark_results.json
```

### 8.2 Plotting
```python
import pandas as pd
import matplotlib.pyplot as plt

# โหลด ablation results
df = pd.read_csv('runs/ablation/ablation_comparison.csv')

# Plot mAP comparison
df.plot(x='experiment', y=['mAP50', 'mAP50-95'], kind='bar')
plt.title('mAP Comparison')
plt.ylabel('mAP')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('runs/ablation/map_comparison.png')

# Plot speed comparison
df.plot(x='experiment', y='fps', kind='bar')
plt.title('FPS Comparison')
plt.ylabel('FPS')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('runs/ablation/fps_comparison.png')
```

---

## 9. Continuous Benchmarking

### 9.1 GitHub Actions (แนะนำ)
สร้าง `.github/workflows/benchmark.yml`:

```yaml
name: Benchmark

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install ultralytics torch torchvision pandas
    
    - name: Run benchmark
      run: |
        python scripts/run_benchmarks.py --mode baseline --device cpu
    
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: runs/baseline/
```

---

## 10. Troubleshooting

### ปัญหาที่พบบ่อย

**Q: ValueError: Dataset not found**
```bash
# ต้องดาวน์โหลด dataset ก่อน
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').val(data='coco128.yaml')"
```

**Q: CUDA out of memory**
```bash
# ลด batch size
python scripts/run_benchmarks.py --batch 1 --device 0

# หรือใช้ CPU
python scripts/run_benchmarks.py --device cpu
```

**Q: numpy version conflict**
```bash
pip install --upgrade numpy
# หรือ
pip install numpy==1.23.5
```

---

## 11. Checklist ก่อน Deploy

```markdown
- [ ] รัน baseline benchmark แล้ว
- [ ] ทดสอบทีละ improvement (ablation study)
- [ ] mAP เพิ่มขึ้น (ระบุ %):
- [ ] FPS ยังอยู่ในเกณฑ์ที่รับได้:
- [ ] Memory usage ไม่เกิน limit:
- [ ] ทดสอบกับ dataset จริงแล้ว
- [ ] Documentation อัปเดตแล้ว
- [ ] Code review เสร็จแล้ว
```

---

## 12. References

- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [COCO Dataset](https://cocodataset.org/)
- [YOLOv8 Paper](https://arxiv.org/abs/2301.05812)

---

## 13. ติดต่อ

หากมีปัญหาหรือคำถาม:
- สร้าง Issue ใน GitHub repository
- ดูไฟล์ `runs/baseline/benchmark_summary.txt` สำหรับผลลัพธ์ล่าสุด
