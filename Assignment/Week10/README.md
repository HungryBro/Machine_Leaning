# Assignment 4: Bayes Decision Theory

ไฟล์ส่งงานมีเพียง `HW4.py` และรันด้วยคำสั่งเดียว:

```bash
python3 HW4.py
```

ค่าเริ่มต้นรันครบ 4 ข้อ และสร้างกราฟ likelihood, posterior และ decision boundary ลงในโฟลเดอร์ `output/`

| ข้อ | ชุดคำสั่งในโค้ด | ขอบตัดสินใจ |
|---:|---|---|
| 1 | `equal_1d`: Gaussian 1D, variance เท่ากัน | single threshold |
| 2 | `unequal_1d`: Gaussian 1D, variance ไม่เท่ากัน | quadratic; อาจมี 0-2 thresholds |
| 3 | `qda_2d`: covariance ต่างกัน | QDA, เส้นโค้ง |
| 4 | `lda_2d`: covariance เดียวกัน | LDA, เส้นตรง |

## ปรับค่าตรงไหน

แก้เฉพาะส่วน **SETTINGS** ด้านบนของ [HW4.py](HW4.py) แล้วรันคำสั่งเดิมอีกครั้ง ไม่ต้องใช้ command options

```python
SCENARIO_TO_RUN = "all"
PARAMETER_MODE = "manual"
N_SAMPLES = 120
RANDOM_SEED = 42
PRIOR_C0 = 0.50
RUN_SENSITIVITY_DEMO = True
```

- `SCENARIO_TO_RUN`: เลือก `"all"`, `"equal_1d"`, `"unequal_1d"`, `"qda_2d"` หรือ `"lda_2d"`
- `PARAMETER_MODE`: `"manual"` ใช้ค่าที่กำหนด; `"estimate"` สุ่มข้อมูล `N_SAMPLES` จุด แล้ว estimate mean, covariance และ prior ด้วย MLE
- `PRIOR_C0`: prior ของ C0 โดย prior ของ C1 คือ `1 - PRIOR_C0`
- `RUN_SENSITIVITY_DEMO`: `True` จะสร้างกราฟทดลองผลของ `n`, `μ`, `σ` และ prior เพิ่ม; ตั้งเป็น `False` หากต้องการเฉพาะกราฟหลัก

## ปรับ μ, σ และ covariance

### กรณี 1D

แก้ใน `ONE_D_PARAMETERS`:

```python
"equal_1d":   {"mu": [-1.5, 1.5], "sigma": [1.0, 1.0]},
"unequal_1d": {"mu": [-1.0, 1.0], "sigma": [0.55, 1.80]},
```

เช่น หากต้องการทดสอบ mean ใกล้กัน:

```python
SCENARIO_TO_RUN = "equal_1d"
ONE_D_PARAMETERS["equal_1d"]["mu"] = [-0.5, 0.5]
RUN_SENSITIVITY_DEMO = False
```

หรือปรับ `N_SAMPLES = 20` แล้วเลือก `PARAMETER_MODE = "estimate"` เพื่อดูความแปรปรวนของ parameter ที่ estimate เมื่อมีข้อมูลน้อย

### กรณี 2D

แก้ใน `TWO_D_PARAMETERS`:

```python
"qda_2d": {
    "mu": [[-1.2, -0.8], [1.1, 0.9]],
    "cov": [[[1.25, 0.55], [0.55, 0.80]],
            [[0.65, -0.35], [-0.35, 1.40]]],
},
```

`mu` แถวแรกคือ C0 และแถวที่สองคือ C1 ส่วน `cov` มี covariance matrix ของ C0 และ C1 ตามลำดับ

สำหรับ LDA ให้ covariance ทั้งสอง matrix เท่ากันเสมอ

## สูตรหลัก

ใช้ Gaussian likelihood:

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

โปรแกรมตัดสินเป็นคลาสที่มี posterior สูงกว่า
