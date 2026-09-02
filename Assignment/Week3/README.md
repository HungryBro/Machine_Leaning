# Week 3 — Bias–Variance Decomposition และ Learning Curves

โค้ดชุดนี้ศึกษาความสัมพันธ์ระหว่าง **bias²**, **variance**, **noise** และ **Eout** โดยเปรียบเทียบโมเดล 3 แบบกับฟังก์ชันเป้าหมาย 2 แบบ

## 1. โจทย์และโมเดล

ข้อมูลฝึกสุ่มจาก \`x ~ Uniform[-1, 1]\` และใช้โมเดลต่อไปนี้:

\`\`\`text
Constant                  g(x) = w0
Linear                    g(x) = w0 + w1 x
Linear through origin     g(x) = w1 x
\`\`\`

ฟังก์ชันเป้าหมายคือ:

\`\`\`text
f(x) = sin(πx)
f(x) = x²
\`\`\`

การ fit ใช้ \`numpy.linalg.lstsq\` ซึ่งให้ผลเทียบเท่าการแก้ least squares/normal equation แต่มีเสถียรภาพกว่าการสร้าง inverse โดยตรง

## 2. สูตรที่ใช้

สำหรับข้อมูลฝึกชุดหนึ่ง \`D\` โมเดลที่ fit ได้คือ \`g_D(x)\` และโมเดลเฉลี่ยคือ:

\`\`\`text
ḡ(x) = E_D[g_D(x)]
\`\`\`

Bias–variance decomposition ที่ใช้ในงานนี้คือ:

\`\`\`text
Eout = Bias² + Variance + σ²
\`\`\`

โดยคำนวณบน \`x ~ Uniform[-1, 1]\` ดังนั้น:

\`\`\`text
E_x[h(x)] = (1/2) ∫[-1,1] h(x) dx
\`\`\`

และ:

\`\`\`text
Bias²    = E_x[(ḡ(x) − f(x))²]
Variance = E_x[E_D[(g_D(x) − ḡ(x))²]]
\`\`\`

## 3. Analytical และ Simulation

### Analytical / numerical integration

ฟังก์ชัน \`analytical_bias_variance()\` ใช้กริดและ numerical integration ด้วย trapezoidal rule:

- สร้างกริด test บน \`[-1, 1]\` จำนวน \`q = 401\` จุด
- พิจารณาคู่ข้อมูลฝึก \`(x₁, x₂)\` บนกริดเดียวกัน
- fit โมเดลแต่ละคู่ข้อมูล แล้วหา \`ḡ(x)\` และ variance
- อินทิเกรต bias² และ variance ตามการแจกแจง Uniform
- Constant model ใช้ค่าคาดหมายแบบปิด ส่วน Linear และ Linear through origin ใช้การอินทิเกรตเชิงตัวเลข

### Simulation

ฟังก์ชัน \`simulate()\` สุ่มข้อมูลฝึก \`n = 2\` จำนวน **50,000 ชุด** แล้ว fit โมเดลบนแต่ละชุด จากนั้นคำนวณ bias², variance และ Eout จากค่าทำนายบน test grid

ค่า simulation ไม่จำเป็นต้องเท่ากับ analytical ทุกหลัก เพราะ simulation เป็นการประมาณจากจำนวนชุดข้อมูลจำกัด แต่ควรใกล้เคียงกัน

## 4. ผลลัพธ์หลักเมื่อ n = 2 และ σ = 0

### Target: \`sin(πx)\`

| Model | Bias² (ana) | Variance (ana) | Eout (ana) | Eout (sim) |
|---|---:|---:|---:|---:|
| Constant | 0.5000 | 0.2500 | 0.7500 | 0.7483 |
| Linear | 0.2067 | 1.6763 | 1.8830 | 1.8835 |
| Linear through origin | 0.2706 | 0.2366 | 0.5072 | 0.5135 |

Linear มี bias ต่ำแต่ variance สูงมาก ส่วน Linear through origin มี Eout ต่ำสุดในชุดการทดลองนี้

### Target: \`x²\`

| Model | Bias² (ana) | Variance (ana) | Eout (ana) | Eout (sim) |
|---|---:|---:|---:|---:|
| Constant | 0.0889 | 0.0444 | 0.1333 | 0.1346 |
| Linear | 0.2000 | 0.3333 | 0.5333 | 0.5342 |
| Linear through origin | 0.2000 | 0.1149 | 0.3149 | 0.3185 |

Constant มี Eout ต่ำสุด เพราะ \`x²\` มีค่าเฉลี่ยคงที่และสมมาตรรอบศูนย์ ขณะที่เส้นตรงไม่สามารถแทนรูปพาราโบลาได้ดี

## 5. Learning curve และ noise

Learning curve ทดลอง \`n = [2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100]\` และ noise 3 ระดับ:

\`\`\`text
σ = 0.0, 0.1, 0.3
y = f(x) + ε,    ε ~ Normal(0, σ²)
\`\`\`

ในกราฟ:

- \`Ein\` คือ MSE บนข้อมูลฝึก
- \`Eout\` คือค่า error บนข้อมูลใหม่ โดยคำนวณ signal error และบวก \`σ²\`
- เมื่อ \`n\` เพิ่ม variance มักลดลงและค่าประมาณนิ่งขึ้น
- เมื่อ \`σ\` เพิ่ม Eout จะสูงขึ้น เพราะมี irreducible noise เพิ่ม

## 6. ไฟล์ผลลัพธ์

\`\`\`text
Week3/
├── bias_variance_lab_compact.py          # โค้ดหลัก
├── bias_variance_lab_compact_comment.py  # wrapper สำหรับรันแบบสั้น
├── README.md
├── Slide/
│   ├── create_slides_week3.py
│   └── slides_Week3.pptx
└── plots/
    ├── average_fit.png
    └── learning_curve.png
\`\`\`

\`average_fit.png\` แสดง target, โมเดลเฉลี่ย และการกระจายของโมเดลจากชุดข้อมูลต่าง ๆ ส่วน \`learning_curve.png\` แสดง \`Ein\` และ \`Eout\` ของทั้ง 3 โมเดลสำหรับ noise ทั้ง 3 ระดับ

## 7. วิธีรัน

รันจากโฟลเดอร์โปรเจกต์:

\`\`\`bash
cd "/Users/dolphin/Desktop/Machine_Learning"
python3 Assignment/Week3/bias_variance_lab_compact.py
\`\`\`

ถ้าต้องการสร้างสไลด์ใหม่หลังจากรันโค้ด:

\`\`\`bash
python3 Assignment/Week3/Slide/create_slides_week3.py
\`\`\`

Dependencies: \`numpy\`, \`matplotlib\`, \`python-pptx\`, \`Pillow\`
