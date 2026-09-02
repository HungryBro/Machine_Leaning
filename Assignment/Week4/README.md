# Week 4 — Performance Estimation

งานนี้เปรียบเทียบวิธีประมาณความผิดพลาดของโมเดลกับค่า **true Eout** ที่คำนวณจากข้อมูลใหม่จำนวนมาก โดยใช้ข้อมูลจากฟังก์ชันเป้าหมาย:

\`\`\`text
f(x) = sin(πx),   x ~ Uniform[-1, 1]
y = f(x) + ε,     ε ~ Normal(0, σ²)
\`\`\`

## 1. โมเดลที่ใช้

โค้ดใช้โมเดล 3 แบบเดียวกันตลอดการทดลอง:

\`\`\`text
Constant                  g(x) = w0
Linear                    g(x) = w0 + w1 x
Linear through origin     g(x) = w1 x
\`\`\`

การ fit ใช้ \`numpy.linalg.lstsq\` และใช้ฟังก์ชัน \`reference_data()\` จาก Week3 เพื่อให้การสร้างข้อมูลอ้างอิงใช้หลักเดียวกัน

## 2. วิธีวัด error

### Resubstitution / Training error

ฝึกโมเดลและวัด MSE บนข้อมูลชุดเดียวกัน:

\`\`\`text
Ein = (1/n) Σ (yi − g(xi))²
\`\`\`

วิธีนี้มักให้ค่าต่ำ เพราะโมเดลได้เห็นข้อมูลที่นำมาวัดแล้ว

### Holdout

แบ่งข้อมูลเป็น train และ test แล้ววัด MSE บนส่วน test โดยทดลองสัดส่วนข้อมูลฝึก:

\`\`\`text
train fraction = 0.1, 0.3, 0.5, 0.7, 0.9
\`\`\`

### K-Fold Cross-Validation

แบ่งข้อมูลเป็น k folds ฝึกด้วย k−1 folds และวัดบน fold ที่เหลือ จากนั้นเฉลี่ย MSE ทุก fold โดยทดลอง:

\`\`\`text
k = 2, 5, 10, 20
\`\`\`

ค่า CV ที่แสดงในตารางอ้างอิงใช้ **5-Fold CV**

## 3. true Eout

สำหรับโมเดลที่ fit จากชุดฝึก ใช้ test grid จำนวน **4,000 จุด** บน \`[-1, 1]\` เพื่อวัด signal error และบวก noise variance:

\`\`\`text
true Eout = mean((f(xtest) − g(xtest))²) + σ²
\`\`\`

ในงานนี้ค่าอ้างอิงชุดเดียวใช้:

\`\`\`text
n = 20
σ = 0.3
seed = 42
σ² = 0.09
\`\`\`

การบวก \`σ²\` สำคัญ เพราะข้อมูลใหม่มี noise เช่นเดียวกับข้อมูลฝึก

## 4. ผลจากข้อมูลอ้างอิงชุดเดียว

| Model | true Eout | Resubstitution | Holdout 70% | 5-Fold CV |
|---|---:|---:|---:|---:|
| Constant | 0.6565 | 0.4264 | 0.4739 | 0.4638 |
| Linear | 0.3003 | 0.1519 | 0.2215 | 0.1848 |
| Linear through origin | 0.2868 | 0.1651 | 0.1128 | 0.1962 |

ข้อสังเกต:

- Resubstitution ต่ำกว่าค่า true Eout ในทุกโมเดลของชุดอ้างอิงนี้
- Holdout และ K-Fold พยายามเลียนแบบ error บนข้อมูลใหม่ จึงมักใกล้ true Eout กว่า resubstitution
- ค่า estimate ในข้อมูลชุดเดียวอาจสูงหรือต่ำกว่า true Eout ได้ เพราะขึ้นกับการสุ่มและการแบ่ง fold
- ค่า true Eout ต่ำสุดในชุดนี้คือ Linear through origin แต่ค่า Holdout ต่ำสุดไม่จำเป็นต้องเลือกโมเดลเดียวกันเสมอไป จึงควรพิจารณาความแปรปรวนและทำซ้ำหลายชุดข้อมูล

## 5. การทดลองหลายชุดข้อมูล

โค้ดทำซ้ำ **2,000 ชุดข้อมูล** แล้วเก็บค่า true Eout และค่าประมาณจากแต่ละวิธี เพื่อดูพฤติกรรมโดยเฉลี่ย:

| Model | Mean true Eout |
|---|---:|
| Constant | 0.6198 |
| Linear | 0.3198 |
| Linear through origin | 0.3055 |

ค่าเฉลี่ยช่วยลดผลจากการสุ่มของข้อมูลชุดเดียว และทำให้เปรียบเทียบ bias/variance ของตัวประมาณได้ชัดขึ้น

## 6. Sensitivity analysis

โค้ดสร้างกราฟ 3 กลุ่ม:

1. \`part2.png\` — bias, variance และ MSE ของ Resubstitution, Holdout และ K-Fold
2. \`part3.png\` — ผลของ train fraction ใน Holdout และจำนวน fold ใน K-Fold
3. \`part4.png\` — ผลของจำนวนข้อมูลและระดับ noise

สำหรับจำนวนข้อมูล โค้ดทดลอง:

\`\`\`text
n = 5, 10, 20, 50, 100
\`\`\`

สำหรับ noise ทดลอง:

\`\`\`text
σ = 0.0, 0.2, 0.4, 0.6, 0.8
\`\`\`

แนวโน้มที่ควรสังเกต:

- เพิ่ม \`n\` ทำให้ค่าประมาณนิ่งขึ้นและ variance ลดลง
- Noise สูงทำให้ true Eout สูงขึ้น เพราะมี \`σ²\` เพิ่ม
- Resubstitution มัก optimistic กว่าวิธีที่ใช้ข้อมูลแยกสำหรับ validation
- Holdout ที่เหลือข้อมูลฝึกน้อยมี variance สูง
- K-Fold ใช้ข้อมูลได้คุ้มกว่า Holdout แต่ค่า estimate ยังขึ้นกับจำนวน fold และชุดข้อมูล

## 7. ไฟล์สำคัญ

\`\`\`text
Week4/
├── HW2.py
├── README.md
├── Slide/
│   ├── create_slides_week4.py
│   └── slides_Week4.pptx
└── plots/
    ├── part2.png
    ├── part3.png
    └── part4.png
\`\`\`

## 8. วิธีรัน

รันจากโฟลเดอร์โปรเจกต์:

\`\`\`bash
cd "/Users/dolphin/Desktop/Machine_Learning"
python3 Assignment/Week4/HW2.py
\`\`\`

ถ้าต้องการสร้างสไลด์ใหม่หลังจากรันโค้ด:

\`\`\`bash
python3 Assignment/Week4/Slide/create_slides_week4.py
\`\`\`

Dependencies: \`numpy\`, \`matplotlib\`, \`python-pptx\`, \`Pillow\`
