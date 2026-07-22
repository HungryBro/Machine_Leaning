# Week 5: อธิบาย HW3.py เทียบกับการคำนวณมือ

ไฟล์หลักของงานคือ [HW3.py](HW3.py)

เอกสารนี้อธิบายโค้ดเทียบกับภาพคำนวณมือจาก 3 กลุ่ม:

- visual_guides/Dregree 3
- visual_guides/Degree 8
- visual_guides/Training,CV

ชื่อ Dregree 3 ใช้ตามชื่อโฟลเดอร์จริง

---

## 1. ภาพรวมของงาน

เป้าหมายคือศึกษาการเลือก degree ของ polynomial และพฤติกรรม overfitting

ฟังก์ชันเป้าหมาย:

\[
f(x)=\sin(\pi x), \qquad -1\le x\le1
\]

ข้อมูลมี 2 แบบ:

Noiseless:

\[
y_i=\sin(\pi x_i)
\]

Noisy:

\[
y_i=\sin(\pi x_i)+\epsilon_i,
\qquad
\epsilon_i\sim N(0,\sigma^2)
\]

โมเดล:

\[
\hat y(x)=w_0+w_1x+w_2x^2+\cdots+w_dx^d
\]

ทดลอง degree:

\[
d=0,1,2,\ldots,8
\]

วัดค่า:

- Training error หรือ \(E_{in}\)
- 10-fold CV error หรือ \(E_{cv}\)
- True \(E_{out}\)
- \(\max |w|\)

ลำดับเดียวกับภาพคำนวณมือคือ:

\[
x,y
\rightarrow X
\rightarrow X^TX,\ X^Ty
\rightarrow w
\rightarrow \hat y
\rightarrow E_{in}
\rightarrow E_{cv}
\rightarrow E_{out}
\]

---

## 2. ข้อมูลจาก Week 3 และ Week 4 ที่นำมาใช้

อยู่ที่ [HW3.py:12-16](HW3.py:12)

    from bias_variance_lab_compact import EOUT_GRID, SEED, reference_data, sin_target
    from HW2 import gen_data

ความหมาย:

| ชื่อ | แหล่งที่มา | หน้าที่ |
|---|---|---|
| sin_target | Week3 | คำนวณ \(\sin(\pi x)\) |
| EOUT_GRID | Week3 | จุด x จำนวนมากสำหรับคำนวณ True Eout |
| reference_data | Week3 | สร้างข้อมูลชุดเดียว |
| SEED | Week3 | ทำให้สุ่มซ้ำได้ |
| gen_data | Week4 | สร้างข้อมูลสุ่มหลายชุดพร้อม noise |

ไม่ได้ import fit_predict จาก Week3 หรือ kfold จาก Week4 โดยตรง เพราะฟังก์ชันเดิมรองรับแค่ Constant, Linear และ Linear through origin

Week5 ต้องรองรับ polynomial degree 0 ถึง D จึงเขียน design, fit และ CV สำหรับ polynomial เพิ่มเอง

---

## 3. ข้อแตกต่างระหว่างข้อมูลมือกับ simulation

ภาพคำนวณมือและ CSV ใช้ x แบบตารางสม่ำเสมอ เช่น n=10:

\[
x=[-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8]
\]

แต่ simulation ใช้การสุ่ม:

\[
x\sim Uniform(-1,1)
\]

ดังนั้นสูตรฟิตเหมือนกัน แต่ข้อมูลตั้งต้นอาจไม่ใช่ชุดเดียวกัน

ถ้าจะเทียบกับภาพคำนวณมือ ให้ดูส่วน supplied CSV

ถ้าจะศึกษาค่าเฉลี่ยและ overfitting ให้ดูส่วน simulation หลาย datasets

---

# ส่วนที่ 1: Polynomial model

## 4. การสร้าง Design Matrix

อยู่ที่ [HW3.py:22-24](HW3.py:22)

    def design(x, d):
        return np.vander(np.asarray(x), d + 1, increasing=True)

สำหรับ degree 3:

\[
X=
\begin{bmatrix}
1&x_1&x_1^2&x_1^3\\
1&x_2&x_2^2&x_2^3\\
\vdots&\vdots&\vdots&\vdots\\
1&x_n&x_n^2&x_n^3
\end{bmatrix}
\]

สำหรับ degree 8:

\[
X=[1,x,x^2,x^3,x^4,x^5,x^6,x^7,x^8]
\]

คำสั่ง increasing=True ทำให้กำลังเรียงจาก 0 ไปถึง d

ดังนั้น:

- degree 3 มี 4 coefficients
- degree 8 มี 9 coefficients
- ถ้า n=10 และ degree 3, X มีขนาด 10×4
- ถ้า n=10 และ degree 8, X มีขนาด 10×9

ภาพในโฟลเดอร์ Dregree 3 แสดง X เป็น [1,x,x²,x³]

ภาพในโฟลเดอร์ Degree 8 แสดง X เป็น [1,x,x²,...,x⁸]

---

## 5. คำนวณ XᵀX และ Xᵀy

อยู่ที่ [HW3.py:27-31](HW3.py:27)

    def fit(x, y, d):
        X = design(x, d)
        XT_X, XT_y = X.T @ X, X.T @ y

ตรงกับสูตรคำนวณมือ:

\[
(X^TX)w=X^Ty
\]

ในภาพมือจะคำนวณ power sums:

\[
S_k=\sum_i x_i^k
\]

และ:

\[
T_j=\sum_i x_i^j y_i
\]

สำหรับ degree 3:

\[
X^TX=
\begin{bmatrix}
S_0&S_1&S_2&S_3\\
S_1&S_2&S_3&S_4\\
S_2&S_3&S_4&S_5\\
S_3&S_4&S_5&S_6
\end{bmatrix}
\]

\[
X^Ty=
\begin{bmatrix}
T_0\\T_1\\T_2\\T_3
\end{bmatrix}
\]

โค้ดไม่ได้เขียน S0, S1, S2 แยกทีละบรรทัด แต่ X.T @ X คำนวณทั้งหมดพร้อมกัน

### Degree 3, noiseless, n=10

จากภาพมือ:

\[
S_0=10,\quad S_1=-1,\quad S_2=3.4,\quad S_3=-1
\]

\[
S_4=2.1328,\quad S_5=-1,\quad S_6=1.62592
\]

จึงได้:

\[
X^TX=
\begin{bmatrix}
10&-1&3.4&-1\\
-1&3.4&-1&2.1328\\
3.4&-1&2.1328&-1\\
-1&2.1328&-1&1.62592
\end{bmatrix}
\]

และ:

\[
X^Ty=
\begin{bmatrix}
0\\
3.077683536\\
0\\
1.143888311
\end{bmatrix}
\]

โค้ดสร้างค่าเดียวกันจากบรรทัด:

    XT_X, XT_y = X.T @ X, X.T @ y

### Degree 3, noiseless, n=80

ภาพมือใช้:

\[
x_i=-1+\frac{2i}{80},\qquad i=0,\ldots,79
\]

และได้ตัวอย่าง power sums:

\[
S_0=80,\quad S_1=-1,\quad S_2=26.675
\]

\[
S_4=16.016665625,\quad S_6=11.4535662207
\]

โค้ดอ่าน x จาก CSV แล้วสร้าง X ใหม่ จึงคำนวณ XᵀX ได้ตามหลักเดียวกัน

### Degree 8

สำหรับ degree 8:

\[
X=[1,x,x^2,\ldots,x^8]
\]

ดังนั้น XᵀX ต้องใช้ power sums ถึง:

\[
S_{16}=\sum x_i^{16}
\]

นี่คือเหตุผลที่ภาพ Degree 8 กล่าวถึง S0 ถึง S16

---

## 6. หา coefficient w

อยู่ที่ [HW3.py:32-36](HW3.py:32)

    w = np.linalg.solve(XT_X, XT_y)

ในกระดาษเขียน:

\[
w=(X^TX)^{-1}X^Ty
\]

โค้ดใช้การแก้ระบบสมการ:

\[
(X^TX)w=X^Ty
\]

ผลทางคณิตศาสตร์เดียวกัน แต่ solve มีความเสถียรกว่าการสร้าง inverse โดยตรง

ถ้าเมทริกซ์มีปัญหา โค้ดใช้ pseudoinverse เป็นทางสำรอง:

    w = np.linalg.pinv(XT_X) @ XT_y

---

# ส่วนที่ 2: เปรียบเทียบ Degree 3 กับภาพมือ

## 7. Degree 3, noiseless, n=10

ค่าจากมือ:

\[
w_0=0.02540358
\]

\[
w_1=2.67394270
\]

\[
w_2=-0.13231034
\]

\[
w_3=-2.86976200
\]

ค่าจากโค้ด:

| coefficient | มือ | โค้ด |
|---|---:|---:|
| w0 | 0.02540358 | 0.025403584895 |
| w1 | 2.67394270 | 2.673942697630 |
| w2 | -0.13231034 | -0.132310337995 |
| w3 | -2.86976200 | -2.869761997863 |

ตรงกันทั้งหมดเมื่อปัดทศนิยม

Training:

\[
MSE_{train}=0.005313778450
\]

\[
RMSE_{train}=0.072895668251
\]

CV:

\[
MSE_{CV}=0.036115025113
\]

\[
RMSE_{CV}=0.190039535658
\]

## 8. Degree 3, noisy, n=10

ภาพมือแสดงค่าประมาณ:

\[
w\approx[0.0108,\ 2.7772,\ 0.5232,\ -2.5980]
\]

ค่าจากโค้ด:

\[
w=[0.010752,\ 2.777216,\ 0.523243,\ -2.597965]
\]

Training RMSE:

\[
0.226547
\]

CV RMSE:

\[
0.428052
\]

CV สูงกว่า Training เพราะโมเดลทดสอบกับข้อมูลที่ไม่ได้ใช้ฟิต และข้อมูลมี noise

## 9. Degree 3, noiseless, n=80

ค่าจากโค้ด:

\[
w=[0.003802,\ 2.691973,\ -0.019022,\ -2.895228]
\]

Training RMSE:

\[
0.066420
\]

CV RMSE:

\[
0.072003
\]

เมื่อ n เพิ่ม coefficient มีความเสถียรมากขึ้น และ CV เข้าใกล้ Training

## 10. Degree 3, noisy, n=80

ค่าจากโค้ด:

\[
w=[-0.075196,\ 2.488371,\ 0.049086,\ -2.746877]
\]

Training RMSE:

\[
0.273691
\]

CV RMSE:

\[
0.290957
\]

ในภาพ Weka บางจุดอ่านค่า w0 เป็น -0.75 แต่ค่าที่ถูกต้องคือประมาณ -0.0752

---

# ส่วนที่ 3: Degree 8 และ Overfitting

## 11. Degree 8, noiseless, n=10

ค่าสัมประสิทธิ์จากโค้ดโดยประมาณ:

\[
w\approx[
0,\ 3.1414,\ 0.0023,\ -5.1630,\ -0.0234,\ 2.5146,\ 0.0699,\ -0.5049,\ -0.0606
]
\]

Training RMSE:

\[
0.000018
\]

CV RMSE:

\[
0.005633
\]

max|w|:

\[
5.1630
\]

ข้อมูลไม่มี noise จึงทำให้ degree 8 ฟิตจุด training ได้เกือบพอดีโดยไม่เกิดความแกว่งรุนแรง

## 12. Degree 8, noisy, n=10

ค่าสัมประสิทธิ์จากโค้ด:

\[
\begin{aligned}
w\approx[&
0.1999,\ 1.0153,\ -5.1034,\ 20.0053,\ 31.2838,\\
&-67.3317,\ -67.4224,\ 55.6246,\ 50.5487]
\end{aligned}
\]

Training RMSE:

\[
0.060663
\]

CV RMSE:

\[
19.042365
\]

max|w|:

\[
67.4224
\]

นี่คือ overfitting ชัดเจน:

- Training ต่ำมาก
- CV สูงมาก
- coefficient มีขนาดใหญ่มาก
- polynomial พยายามตาม noise ของข้อมูล 10 จุด

## 13. Degree 8, noiseless, n=80

ค่าสัมประสิทธิ์จากโค้ดโดยประมาณ:

\[
w\approx[
0,\ 3.1399,\ 0.0009,\ -5.1425,\ -0.0058,\ 2.4489,\ 0.0117,\ -0.4471,\ -0.0071
]
\]

Training RMSE:

\[
0.000153
\]

CV RMSE:

\[
0.000224
\]

เมื่อข้อมูลสะอาดและมีจำนวนมาก degree 8 ยังทำงานได้ดี

## 14. Degree 8, noisy, n=80

ค่าสัมประสิทธิ์จากโค้ด:

\[
w\approx[
-0.1266,\ 3.1207,\ 1.7580,\ -6.4256,\ -8.0255,\ 5.2356,\ 11.6030,\ -1.9941,\ -5.1145
]
\]

Training RMSE:

\[
0.257766
\]

CV RMSE:

\[
0.297913
\]

max|w|:

\[
11.6030
\]

ยังมีความไม่เสถียร แต่รุนแรงน้อยกว่า noisy n=10

สรุป:

\[
n\uparrow \Rightarrow \text{coefficient แกว่งน้อยลง}
\]

---

# ส่วนที่ 4: Prediction และ MSE

## 15. การคำนวณ prediction

อยู่ที่ [HW3.py:39-42](HW3.py:39)

    def predict(x, w):
        y = design(x, len(w) - 1) @ w
        return y

ตรงกับ:

\[
\hat y=Xw
\]

สำหรับ degree 3:

\[
\hat y=w_0+w_1x+w_2x^2+w_3x^3
\]

สำหรับ degree 8:

\[
\hat y=w_0+w_1x+\cdots+w_8x^8
\]

ฟังก์ชัน predict ถูกใช้ทั้ง Training, CV และ Eout

## 16. การคำนวณ Training MSE

อยู่ที่ [HW3.py:45-47](HW3.py:45)

    def mse(y, yhat):
        return float(np.mean(np.square(yhat - y)))

ตรงกับ:

\[
MSE=\frac{1}{n}\sum_i(y_i-\hat y_i)^2
\]

โค้ดใช้ np.mean จึงรวมการบวกและการหาร n ในครั้งเดียว

ถ้าต้องการ RMSE:

\[
RMSE=\sqrt{MSE}
\]

ตัวอย่าง Degree 3 noiseless n=10:

\[
MSE=0.0053138
\]

\[
RMSE=0.0728957
\]

---

# ส่วนที่ 5: 10-fold Cross-validation

## 17. การแบ่ง Train/Test

อยู่ที่ [HW3.py:71-80](HW3.py:71)

สำหรับ k=10:

| n | Train ต่อรอบ | Test ต่อรอบ | จำนวนรอบ |
|---:|---:|---:|---:|
| 10 | 9 | 1 | 10 |
| 20 | 18 | 2 | 10 |
| 40 | 36 | 4 | 10 |
| 80 | 72 | 8 | 10 |

แต่ละรอบใช้ 9 folds train และ 1 fold test

เมื่อครบ 10 รอบ ข้อมูลทุกจุดจะถูกใช้เป็น test หนึ่งครั้ง

## 18. การคำนวณ CV

อยู่ที่ [HW3.py:83-89](HW3.py:83)

    def cv_mse(x, y, d, folds):
        errors = []
        all_idx = np.arange(len(x))

        for test in folds:
            train = np.setdiff1d(all_idx, test)
            w = fit(x[train], y[train], d)
            prediction = predict(x[test], w)
            errors.extend(np.square(prediction - y[test]))

        return float(np.mean(errors))

ตรงกับภาพมือในโฟลเดอร์ Dregree 3:

1. เลือก test fold
2. ใช้ fold อื่นเป็น train
3. หา w ใหม่จาก train
4. ทำนาย test
5. หาความคลาดเคลื่อน
6. ทำครบ 10 รอบ
7. เฉลี่ย squared error ทั้งหมด

สำหรับ Degree 3 noiseless n=10:

\[
MSE_{CV}=0.036115025113
\]

\[
RMSE_{CV}=0.190039535658
\]

---

# ส่วนที่ 6: Weka-like folds

## 19. JavaRandom

อยู่ที่ [HW3.py:50-68](HW3.py:50)

    class JavaRandom:
        """Java/Weka Random(seed)"""
        ...

และถูกใช้ใน:

    evaluate(..., weka=True)

เหตุผลคือ Weka ใช้การสุ่มแบบ Java Random จึงเขียน JavaRandom เพื่อให้การเรียงลำดับข้อมูลและการแบ่ง fold ใกล้เคียง Weka

ส่วนข้อมูล CSV อยู่ที่ [HW3.py:125-148](HW3.py:125)

    a = np.genfromtxt(...)
    x, y = a[a.dtype.names[0]], a[a.dtype.names[-1]]
    table, weights = evaluate(..., weka=True)

ค่า Weka ไม่ได้ถูก hard-code ในโค้ด

---

# ส่วนที่ 7: Training,CV และการเลือก Degree

## 20. evaluate()

อยู่ที่ [HW3.py:97-105](HW3.py:97)

    for d in range(D + 1):
        w = fit(x, y, d)
        rows.append([
            mse(y, predict(x, w)),
            cv_mse(x, y, d, folds),
            true_eout(w, sigma),
            np.max(np.abs(w))
        ])

ในแต่ละ degree จะเก็บ:

| ตำแหน่ง | ความหมาย |
|---:|---|
| row[0] | Training MSE หรือ Ein |
| row[1] | CV MSE |
| row[2] | True Eout |
| row[3] | max|w| |

ถ้า D=8 จะวน d=0 ถึง 8

## 21. ผลเฉลี่ยจาก simulation 300 datasets

ค่าจาก n=20, sigma=0.3:

| d | Mean Ein | Mean Ecv | Mean Eout |
|---:|---:|---:|---:|
| 0 | 0.5645 | 0.6262 | 0.6144 |
| 1 | 0.2540 | 0.3249 | 0.3179 |
| 2 | 0.2312 | 0.4004 | 0.3667 |
| 3 | 0.0734 | 0.1316 | 0.1245 |
| 4 | 0.0683 | 0.1904 | 0.1819 |
| 5 | 0.0625 | 0.6067 | 0.4669 |
| 6 | 0.0585 | 3.4640 | 1.2954 |
| 7 | 0.0540 | 350.5220 | 13.0970 |
| 8 | 0.0491 | 2748.0414 | 139.3312 |

ผลที่ได้:

- Training error เลือก d=8
- CV เลือก d=3
- True Eout เลือก d=3

เหตุผลคือ Training วัดข้อมูลที่โมเดลเคยเห็นแล้ว จึงชอบโมเดลซับซ้อน

แต่ CV วัดข้อมูลที่ถูกกันออกจาก training จึงเห็น overfitting

---

# ส่วนที่ 8: True Eout

## 22. สูตร True Eout

อยู่ที่ [HW3.py:92-94](HW3.py:92)

    def true_eout(w, sigma):
        return mse(
            sin_target(EOUT_GRID),
            predict(EOUT_GRID, w)
        ) + sigma ** 2

ตรงกับ:

\[
E_{out}
\approx
\frac{1}{N}\sum
(\hat y(x)-\sin(\pi x))^2+\sigma^2
\]

โดย:

- EOUT_GRID คือ x จำนวนมากบนช่วง [-1,1]
- sin_target คือฟังก์ชันจริง
- predict คือ polynomial ที่ fit ได้
- sigma² คือ noise variance

ถ้า noiseless:

\[
\sigma=0
\]

ถ้า noisy และ sigma=0.3:

\[
\sigma^2=0.09
\]

ภาพมือเน้น Training และ CV เป็นหลัก ส่วน True Eout เป็นส่วนที่โค้ดเพิ่มเพื่อให้ครบตามโจทย์ simulation

---

# ส่วนที่ 9: Simulation หลายชุดข้อมูล

## 23. simulate()

อยู่ที่ [HW3.py:108-114](HW3.py:108)

    def simulate(n, sigma, D, k, reps, seed):
        np.random.seed(seed)
        out = []

        for r in range(reps):
            x, y = gen_data(n, sigma)
            out.append(evaluate(x, y, D, k, sigma, seed + r)[0])

        return np.asarray(out)

การทำงาน:

1. สุ่ม dataset ใหม่
2. Fit ทุก degree
3. เก็บ Ein, Ecv, Eout และ max|w|
4. ทำซ้ำตาม reps
5. หาค่าเฉลี่ยราย degree

ใน main:

    runs = simulate(...)
    mean = runs.mean(0)

จากนั้นคำนวณ frequency ว่าแต่ละวิธีเลือก degree ไหน

---

# ส่วนที่ 10: ผลของ n และ sigma

## 24. sensitivity experiment

อยู่ที่ [HW3.py:220-237](HW3.py:220)

ทดลอง:

\[
n\in\{10,20,40,80\}
\]

\[
\sigma\in\{0,0.3,0.6\}
\]

เก็บค่า:

- degree ที่ Training เลือก
- degree ที่ CV เลือก
- degree ที่ Eout เลือก
- gap@D
- median max|w| ที่ degree สูงสุด

นิยาม:

\[
gap@D=Eout(D)-Ein(D)
\]

ถ้า gap ใหญ่ แปลว่าโมเดลทำได้ดีบน training แต่แย่บนข้อมูลใหม่

ตัวอย่าง noisy sigma=0.3:

| n | CV เลือก degree | median max|w| ที่ d=8 |
|---:|---:|---:|
| 10 | 1 | 609.94 |
| 20 | 3 | 51.63 |
| 40 | 3 | 21.21 |
| 80 | 3 | 12.20 |

สรุป:

\[
n\uparrow \Rightarrow \text{overfitting ลดลง}
\]

และ:

\[
\sigma\uparrow \Rightarrow \text{overfitting รุนแรงขึ้น}
\]

---

# ส่วนที่ 11: max|w| กับ overfitting

## 25. การเก็บขนาด coefficient

อยู่ที่ [HW3.py:103-104](HW3.py:103)

    np.max(np.abs(w))

ตรงกับ:

\[
\max|w|=\max(|w_0|,\ldots,|w_d|)
\]

ตัวอย่าง:

| กรณี | max|w| |
|---|---:|
| Degree 8, noiseless n=10 | 5.163 |
| Degree 8, noisy n=10 | 67.422 |
| Degree 8, noisy n=80 | 11.603 |

Degree 8 noisy n=10 มี coefficient ใหญ่ที่สุด เพราะข้อมูลมีน้อยและมี noise มากพอที่ polynomial จะพยายามตาม noise

---

# ส่วนที่ 12: MSE กับ RMSE

ในโค้ดหลักใช้ MSE:

\[
MSE=\frac{1}{n}\sum(y-\hat y)^2
\]

แต่ตาราง Weka และภาพ Training,CV มักแสดง RMSE:

\[
RMSE=\sqrt{MSE}
\]

ดังนั้นก่อนเทียบต้องดูชื่อ metric ก่อน

ตัวอย่าง:

\[
RMSE=0.2265
\]

จะเท่ากับ:

\[
MSE=0.2265^2\approx0.0513
\]

ไฟล์ provided_data_metrics.csv เก็บทั้ง:

- train_mse
- train_rmse
- cv_mse
- cv_rmse

---

# ส่วนที่ 13: จุดที่ตัวเลข Weka อาจต่างจากโค้ด

## Degree 3 noisy n=80

ภาพ Weka บางภาพอ่าน w0 เป็น -0.75 แต่ค่าที่ถูกต้องจากโค้ดคือ:

\[
w_0=-0.0752
\]

## Degree 8 coefficient order

ภาพ Weka บางภาพเรียงแถวเป็น w1 ถึง w8 แล้วค่อย w0

แต่ไฟล์โค้ดเรียงตามลำดับที่ถูกต้อง:

\[
w_0,w_1,\ldots,w_8
\]

## Degree 8 noisy n=10 CV

ภาพ Weka ประมาณ:

\[
CV\ RMSE\approx18.97
\]

โค้ดจาก CSV ปัจจุบัน:

\[
CV\ RMSE=19.0424
\]

## Degree 8 noiseless n=10 CV

โค้ดได้:

\[
CV\ RMSE=0.0056
\]

ความแตกต่างอาจเกิดจาก:

- การปัดเศษ y ใน CSV
- Weka ใช้ข้อมูลก่อนปัดเศษ
- วิธีแก้ระบบสมการแตกต่างกัน
- fold หรือ setting ของ Weka แตกต่างกัน

ไม่ควร hard-code ค่า Weka เพื่อบังคับให้ตรง ควรใช้สูตรเดียวกันและรายงานความแตกต่างที่เกิดจาก precision/implementation

---

# สรุปสำหรับเขียนรายงาน

1. โค้ดสร้าง Design Matrix \([1,x,\ldots,x^d]\) เหมือนการคำนวณมือ
2. X.T @ X ตรงกับการรวม power sums \(S_k\)
3. X.T @ y ตรงกับ \(T_j=\sum x_i^jy_i\)
4. np.linalg.solve ตรงกับการแก้ \(w=(X^TX)^{-1}X^Ty\)
5. Degree 3 noiseless n=10 ให้ coefficient ตรงกับภาพมือ
6. Degree 8 ใช้ coefficient 9 ตัวและต้องรวม power sum ถึง S16
7. Training error มักลดลงเมื่อ degree เพิ่ม
8. CV และ True Eout ลดลงในช่วงแรก แล้วเพิ่มเมื่อ degree สูงเกินไป
9. Degree 8 noisy n=10 overfit รุนแรงที่สุด
10. เพิ่ม n ช่วยลดความผันผวนของ coefficient
11. เพิ่ม sigma ทำให้ high-degree model overfit มากขึ้น
12. max|w| เป็นสัญญาณประกอบของความไม่เสถียร
13. Simulation โดยเฉลี่ยเลือก d=3 จาก CV และ True Eout
14. Training error เลือก d=8 เพราะโมเดลซับซ้อนสามารถลด error บนข้อมูลเดิมได้มาก

ข้อสรุปหลัก:

\[
\boxed{
\text{ใช้ CV เลือก degree แทนการเลือกจาก Training error เพียงอย่างเดียว}
}
\]

หลังเลือก degree แล้วจึง fit โมเดลสุดท้ายด้วยข้อมูลทั้งหมด

