# Assignment 4: Bayes Decision Theory

ไฟล์ส่งงานมีเพียง `HW4.py` และรันด้วยคำสั่งเดียว:

```bash
python3 HW4.py
```

โปรแกรมจะรันครบ 4 ข้อ สร้างกราฟ และบันทึกผลไว้ในโฟลเดอร์ `output/` โดยอัตโนมัติ

| ข้อ | กรณี | ตัวจำแนก | สิ่งที่กราฟแสดง |
|---:|---|---|---|
| 1 | Gaussian 1 ตัวแปร, variance สองคลาสเท่ากัน | Single Threshold | likelihood, posterior, threshold/decision regions |
| 2 | Gaussian 1 ตัวแปร, variance สองคลาสไม่เท่ากัน | Quadratic / Double Threshold | likelihood, posterior, 0-2 thresholds |
| 3 | Gaussian 2 ตัวแปร, covariance ต่างกัน | QDA | likelihood contours, posterior, quadratic decision curve |
| 4 | Gaussian 2 ตัวแปร, covariance ร่วมกัน | LDA | likelihood contours, posterior, linear decision line |

## ปรับค่าอย่างไร

ไม่ต้องใส่ options ต่อท้ายคำสั่งรัน ให้แก้ไขเฉพาะส่วน **SETTINGS: ปรับค่าตรงนี้** ที่ด้านบนของ `HW4.py` แล้วรันคำสั่งเดิมอีกครั้ง

```python
SCENARIO_TO_RUN = "all"
PARAMETER_MODE = "manual"
N_SAMPLES = 120
RANDOM_SEED = 42
PRIOR_C0 = 0.50

MU_C0 = None
MU_C1 = None
SIGMA_C0 = None
SIGMA_C1 = None
COV_C0 = None
COV_C1 = None

RUN_SENSITIVITY_DEMO = True
```

### ความหมายของตัวแปรตั้งค่า

| ตัวแปร | ความหมาย |
|---|---|
| `SCENARIO_TO_RUN` | `"all"` เพื่อรันครบทุกข้อ หรือเลือก `"equal_1d"`, `"unequal_1d"`, `"qda_2d"`, `"lda_2d"` |
| `PARAMETER_MODE` | `"manual"` ใช้พารามิเตอร์ที่กำหนดเอง; `"estimate"` สุ่มข้อมูลแล้ว estimate พารามิเตอร์แบบ MLE |
| `N_SAMPLES` | จำนวนตัวอย่างที่สุ่มในโหมด `estimate` |
| `PRIOR_C0` | Prior \(P(C_0)\); ส่วน \(P(C_1)=1-P(C_0)\) |
| `MU_C0, MU_C1` | ค่าเฉลี่ยของ C0 และ C1 |
| `SIGMA_C0, SIGMA_C1` | ส่วนเบี่ยงเบนมาตรฐานของกรณี 1D |
| `COV_C0, COV_C1` | covariance ของกรณี 2D |
| `RUN_SENSITIVITY_DEMO` | `True` สร้างกราฟทดลองปรับค่าเพิ่ม; `False` สร้างเฉพาะกรณีที่เลือก |

หากค่า mean หรือ covariance เป็น `None` โปรแกรมจะใช้ค่าตั้งต้นที่ตั้งใจให้เห็น decision boundary ชัดเจน

## ตัวอย่างการทดลอง

### 1) ดูผลของจำนวนตัวอย่าง n

แก้ SETTINGS เป็น:

```python
SCENARIO_TO_RUN = "equal_1d"
PARAMETER_MODE = "estimate"
N_SAMPLES = 20
RUN_SENSITIVITY_DEMO = False
```

รันหนึ่งครั้ง จากนั้นเปลี่ยน `N_SAMPLES = 500` แล้วรันซ้ำ จะเห็นว่า parameter ที่ estimate และตำแหน่ง threshold นิ่งขึ้นเมื่อจำนวนข้อมูลมากขึ้น

### 2) ดูผลของค่าเฉลี่ย μ

```python
SCENARIO_TO_RUN = "equal_1d"
PARAMETER_MODE = "manual"
MU_C0 = "-0.5"
MU_C1 = "0.5"
SIGMA_C0 = 1.0
SIGMA_C1 = 1.0
RUN_SENSITIVITY_DEMO = False
```

ค่า mean ที่ใกล้กันทำให้ likelihood ซ้อนทับมากและ accuracy ลดลง ลองเปลี่ยนเป็น `MU_C0 = "-2.0"` กับ `MU_C1 = "2.0"` แล้วรันอีกครั้งเพื่อเปรียบเทียบ

### 3) ดูผลของ σ

```python
SCENARIO_TO_RUN = "unequal_1d"
PARAMETER_MODE = "manual"
MU_C0 = "-1.0"
MU_C1 = "1.0"
SIGMA_C0 = 0.55
SIGMA_C1 = 1.80
RUN_SENSITIVITY_DEMO = False
```

เมื่อ sigma ไม่เท่ากัน decision boundary อาจมี 0, 1 หรือ 2 thresholds ตามพารามิเตอร์

### 4) ดูผลของ prior ใน LDA

```python
SCENARIO_TO_RUN = "lda_2d"
PARAMETER_MODE = "manual"
PRIOR_C0 = 0.80
RUN_SENSITIVITY_DEMO = False
```

decision region ของ C0 จะขยาย และเส้น boundary จะเลื่อนไปทาง C1 ซึ่งมี prior น้อยกว่า

### 5) กำหนด mean/covariance ของ QDA แบบ 2D

```python
SCENARIO_TO_RUN = "qda_2d"
MU_C0 = "-1.4,-0.8"
MU_C1 = "1.0,1.2"
COV_C0 = "1.2,0.5,0.5,0.8"
COV_C1 = "0.6,-0.3,-0.3,1.5"
RUN_SENSITIVITY_DEMO = False
```

covariance 2D ใช้รูปแบบ `"a,b,c,d"` ซึ่งหมายถึง

\[
\begin{bmatrix}a&b\\c&d\end{bmatrix}.
\]

## สูตรที่ใช้

สำหรับแต่ละคลาส \(C_k\) ใช้ Gaussian likelihood:

\[
p(\mathbf{x}\mid C_k)=
\frac{1}{(2\pi)^{d/2}|\Sigma_k|^{1/2}}
\exp\left[-\frac{1}{2}(\mathbf{x}-\mu_k)^T\Sigma_k^{-1}(\mathbf{x}-\mu_k)\right]
\]

และ Bayes posterior:

\[
P(C_k\mid\mathbf{x})=
\frac{p(\mathbf{x}\mid C_k)P(C_k)}
{\sum_j p(\mathbf{x}\mid C_j)P(C_j)}.
\]

- ข้อ 1: variance เท่ากัน จึงมี single threshold
- ข้อ 2: variance ไม่เท่ากัน จึงเป็น quadratic และอาจได้ two thresholds
- ข้อ 3: covariance ต่างกัน จึงเป็น QDA ที่ boundary เป็นเส้นโค้ง
- ข้อ 4: covariance เดียวกัน จึงเป็น LDA ที่ boundary เป็นเส้นตรง
