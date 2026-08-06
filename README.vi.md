# Dự đoán địa chất dọc theo giếng khoan ngang

*[English](README.md) · [Tiếng Việt](README.vi.md)*

Giải pháp cho cuộc thi ROGII Wellbore Geology Prediction: dự đoán **TVT**
(true vertical thickness — vị trí địa tầng của giếng bên trong cột đá) cho phần
chưa được đo log của một giếng khoan ngang.

**Kết quả: 10.150 public / 9.402 private RMSE — hạng 980 / 6191 (top 16%).**

| Giai đoạn | Private RMSE |
|---|---|
| Baseline giữ nguyên giá trị cuối (`TVT = giá trị biết cuối cùng`) | ~15.9 |
| Bài nộp đầu tiên (particle filter) | 15.386 |
| GBDT stack trên feature tự xây | 13.565 |
| + prior mặt cấu trúc, DP tracker, leak guard | 12.022 |
| **Cuối: hai tracker độc lập + stack + projection** | **9.402** |

---

## Một phương trình là nền tảng của tất cả

Trên mọi giếng train đã kiểm tra, đúng chính xác đến ±0.05 ft (bằng đúng độ làm
tròn của dữ liệu gốc):

```
TVT  =  -Z  +  ANCC(x, y)  +  c_well
```

* `std(TVT + Z - ANCC)` mỗi giếng: median **0.0066 ft**, max **0.0079 ft** (150 giếng)
* Sáu cột formation chỉ là hằng số cộng của nhau *trong cùng một giếng*:
  `std(ASTNU - ANCC)` ≈ **1e-13**

`Z` được cho sẵn trong tập test. Vậy dự đoán TVT **chính là** dự đoán một mặt
cấu trúc dọc theo giếng — không gì khác. Mọi feature trong `src/geosteer.py` đều
là một cách ước lượng khác nhau của cùng đại lượng đó.

Cách nhìn này là thứ mở khoá cả bài toán. Nó cũng loại bỏ ngay một sai lầm hiển
nhiên: tóm tắt độ sâu formation của một giếng bằng **median** là vứt bỏ toàn bộ
tín hiệu, vì mặt cấu trúc dốc khoảng 150 ft dọc theo *một* lateral duy nhất. Dự
đoán bằng `ANCC` hằng số cho RMSE 106–109 trong khi baseline là 15.2.

## Bài toán khó tới đâu? Hai thang oracle

**Thang 1 — hiệu chỉnh theo giếng có thể tốt tới đâu?** (250 giếng, fit oracle)

| Dự đoán | Pooled RMSE |
|---|---|
| Baseline giữ giá trị cuối | 16.61 |
| Oracle: **hằng số** tốt nhất mỗi giếng | **9.07** |
| Oracle: đường thẳng tốt nhất mỗi giếng | 6.59 |
| Oracle: đường cong bậc 6 tốt nhất mỗi giếng | 3.00 |

Ước lượng *mức* hoàn hảo cũng chỉ dừng ở 9.07. Muốn xuống thấp hơn bắt buộc phải
bám **hình dạng**. Bảng này định hình toàn bộ chiến lược: feature mặt cấu trúc lo
phần mức, GR tracker lo phần hình dạng.

**Thang 2 — mô tả một giếng cần bao nhiêu con số?** Fit đa thức robust cho mức
địa tầng `U = TVT + Z` theo MD chuẩn hoá:

| Bậc | RMSE |
|---|---|
| 2 | 3.98 |
| 3 | 3.10 |
| 4 | 2.32 |
| 6 | 1.71 |

**Một đa thức bậc ba theo MD chuẩn hoá mô tả trọn quỹ đạo vùng eval tới ~3 ft.**
Đáp án của một giếng về cơ bản chỉ là bốn con số. Đó chính là thứ lớp projection
ở mục 7 khai thác.

---

## Kiến trúc

```
                    ┌─ chuyển giao mặt cấu trúc ─────┐
                    │   (giếng lân cận, ~300 ft)     │
feature theo dòng ──┼─ DP (TVT, dip) Viterbi ────────┼──► LGBM ×3 + CatBoost ×2
                    │   tất định, toàn cục           │         │
                    ├─ particle filter theo like ────┤         ▼
                    │   ngẫu nhiên, tiến             │    Ridge stack
                    └─ landscape sai lệch GR ────────┘         │
                        (11 offset × 4 mốc)                     ▼
                                                    trộn với tracker
                                                            ▼
                                                    projection bậc ba của U
                                                            ▼
                                                    co về baseline
```

### Ba cách ước lượng mặt cấu trúc

**1. Chuyển giao mặt cấu trúc.** Các lateral lân cận lấy mẫu cùng một mặt —
khoảng cách trung vị tới giếng gần nhất chỉ **~300 ft**. Với giếng lân cận *j*:
`TVT_j(i) = -Z(i) + ANCC_j(điểm gần nhất của j) + o_j`, trong đó `o_j` được fit
trên phần lateral đã biết của giếng đích. Mốc neo đó hấp thụ cả chênh lệch datum
*lẫn* độ lệch ngang gần như không đổi giữa hai lateral song song. Các giếng lân
cận được đánh trọng số theo sai số trên một lát **giữ riêng** của vùng đã biết,
nên trọng số là trung thực.

Chỉ dùng một formation: cả sáu sụp về cùng một mảng khi `o_j` đã hấp thụ hằng số
chênh lệch (spread đo được **7.7e-5 ft**). Spread có ý nghĩa là spread giữa các
*giếng lân cận*, không phải giữa các formation.

**2. DP trên (TVT, dip địa tầng).** Viterbi với dip là *state tường minh*, nên
đường đi là tuyến tính từng khúc theo dip — vài đoạn cho mỗi lateral, đúng cách
một người phiên giải thực sự làm. Beam search không có state như vậy, đó là lý do
hai cái không dư thừa lẫn nhau.

**3. Particle filter có trọng số likelihood.** State là `U = TVT + Z` với tốc độ
random-walk; mỗi trong 16 seed được chấm bằng tổng log-likelihood rồi kết hợp
bằng softmax ở nhiều nhiệt độ. Vector hoá theo (seed × particle) nên không cần
numba.

### Vì sao dùng hai GR tracker thay vì một

Đo độc lập, không có model nào tham gia:

| | Bank 481 giếng (45 giếng) | Bank 773 giếng (30 giếng) |
|---|---|---|
| Baseline giữ giá trị cuối | 16.168 | 18.384 |
| Chuyển giao mặt cấu trúc | 18.117 | **14.110** |
| DP (TVT, dip) | 12.785 | 14.967 |
| PF theo likelihood | **12.031** | 15.817 |
| **50/50 DP + PF** | **10.587** | **12.415** |

Tương quan sai số theo giếng giữa DP và PF: **0.095–0.258**. Chúng hỏng ở những
chỗ hoàn toàn khác nhau:

| Giếng | Baseline | Surf | DP | PF |
|---|---|---|---|---|
| fef8af96 | 44.6 | **4.0** | 18.6 | 49.5 |
| 6d6d93af | 32.2 | **6.1** | 14.3 | 54.7 |
| 197f8a5a | 24.6 | 48.5 | 45.9 | **5.1** |
| a8ed028a | 19.0 | 27.1 | 20.7 | **4.1** |

Trung bình cộng ăn hơn tracker tốt nhất khoảng 1.5 ft. **Lưu ý hai cột mâu thuẫn
nhau về việc tracker nào tốt nhất** — prior mặt cấu trúc cần giếng lân cận dày,
nên chất lượng của nó phụ thuộc số giếng trong bank. Vì vậy notebook *sweep*
trọng số trộn thay vì hardcode.

### Sai lệch GR được đưa vào nguyên vẹn, không phải dạng argmin

Sai lệch GR tại đường đi đúng thấp hơn ~3 lần so với tại một offset hằng số sai:

| Offset δ (ft) | −20 | −10 | **0** | +10 | +20 |
|---|---|---|---|---|---|
| Sai lệch chuẩn hoá | 3.20 | 2.69 | **1.05** | 3.13 | 3.48 |

18 trên 20 giếng có cực tiểu tại δ = 0. Nhưng landscape đa cực trị nặng — quét
offset hằng số thường xuyên tìm thấy cực tiểu giả tại ±30 ft với chi phí gần
bằng đường đúng. Nên thay vì cam kết vào một argmin, các feature `tda*`, `tdbc*`,
`tddp*`, `tdsf*`, `tdpf*` đưa cho model sai lệch tại 11 offset quanh mỗi mốc và
để model tự chọn mode.

---

## Kết quả

Leave-one-well-out CV, 160 giếng, GroupKFold, một sklearn HGB:

| Bước | Pooled RMSE |
|---|---|
| Baseline giữ giá trị cuối | 15.192 |
| Feature set v1 | 13.312 |
| **+ sửa bug lateral** | **12.920** |
| + trộn tracker + projection bậc ba + shrink | **12.161** |

Chênh lệch giữa các fold rất lớn (10.29 → 15.43) và fold 3 vẫn thua baseline của
chính nó. Pooled RMSE bị chi phối bởi một nhúm giếng drift.

---

## Hai bug đáng đọc

Cả hai cùng một nguyên nhân gốc: **dùng toàn bộ vùng đã biết thay vì chỉ phần
lateral của nó.** Vùng đã biết chứa build section, nơi TVT dao động hàng trăm ft.

| Feature | Trước | Sau |
|---|---|---|
| `sc_ens_d` (khớp cửa sổ NCC) | median **−92 ft**, 26% dòng vượt ±200, max 1155 | median 1.26, max 8.7 |
| `slp_b_d` (ngoại suy tuyến tính) | median **+945 ft**, max 4268 | median −28.9, max 57.8 |

NCC đang khớp cửa sổ GR vùng eval với cửa sổ trong *build section*; phần fit độ
dốc đọc dip từ đường build rồi ngoại suy 4000 ft. Sửa cả hai đáng **0.39 ft**
pooled — nhiều hơn phần lớn các thay đổi mô hình.

Không cái nào lộ ra trong RMSE tổng. Thứ bắt được chúng là kiểm tra khoảng giá
trị feature (mục 5 của notebook): mọi tracker feature đều là delta TVT, mà
`TVT − giá trị biết cuối` có p99 ≈ 56 ft và absmax ≈ 84 ft trên toàn tập train,
nên **bất cứ giá trị nào vượt ±100 đều vô nghĩa theo định nghĩa**. Kiểm tra đó
giờ là một cell cố định.

---

## Kết quả âm

Ghi lại vì chúng tốn thời gian thật và lý do thất bại có tính tổng quát.

### Cổng tin cậy theo từng giếng không hoạt động

Pooled RMSE bị chi phối bởi vài giếng drift, nên biết được *theo từng giếng* nên
tin model tới đâu trông như đòn bẩy hiển nhiên. Headroom là có thật:

| | Pooled RMSE |
|---|---|
| Model | 13.312 |
| + shrink toàn cục | 13.020 |
| Oracle: chọn model hoặc baseline mỗi giếng | 11.21 |
| Oracle: shrink tối ưu mỗi giếng | 10.00 |

Nhưng không có gì quan sát được dự đoán nổi nó. Tương quan giữa shrink tối ưu
theo giếng và mọi thống kê sẵn có đạt đỉnh ở **0.148** (`beam_std_d`). Một
backtest chỉ trên lateral qua 138 giếng cũng không khá hơn (|corr| tốt nhất
**0.164**).

Nguyên nhân mang tính cấu trúc. Phần lateral đã biết chỉ dài 350–650 dòng (đo
được: 345, 613, 650, 542, 352, 604), nên nhiều nhất che được ~250 dòng — khoảng
250 ft so với vùng eval 4000–6000 ft. Trên 250 ft thì baseline đã gần như chính
xác: `r_base` self-check trung bình là **2.33 ft** so với **15.19 ft** trên vùng
eval thật. Backtest đang đo một chế độ hoàn toàn khác nên không có khả năng phân
biệt. Không sửa được bằng thiết kế tốt hơn.

### Bộ làm mượt Bayesian toàn cục thua cả việc không làm gì

Bộ làm mượt tiến-lùi trên `U` đạt **24.58** so với baseline **16.16**, với 10%
giếng tệ nhất tạo ra 56% tổng bình phương sai số. Shrink và clip kéo về 14.40,
nhưng lợi ích so với stack chỉ +0.05 ft — dưới mức nhiễu seed. Cùng kiểu thất bại
với DP không ràng buộc: thiếu prior mạnh thì likelihood GR đi lạc vào cực tiểu
giả.

### Không đáng khai thác phần trùng train/test

Cuộc thi để lọt ba giếng test hiển thị vào luôn trong `train/`, kèm `TVT` và các
cột formation. Các lời giải tham khảo tái dựng TVT từ contact formation cho những
giếng đó và báo sai số gần bằng không.

**Đó là code chết trên tập được chấm.** Một bài nộp probe cộng +100 vào mọi dòng
và +1000 vào các giếng bị override trả về **100.665** — nếu tập được chấm chính
là các giếng đó thì điểm phải khoảng 1100. Public leaderboard chạy trên giếng
khác.

Repo này loại bỏ nó có chủ đích, và chặn theo hình học chứ không theo well id:

```python
geometric_exclusions(h, bank, min_dist=50.0)
```

Mọi giếng trong bank có quỹ đạo nằm trong 50 ft của giếng đích đều bị loại. Phân
tách rất rõ ràng — bản sao của chính giếng nằm ở ~1 ft, lateral lân cận thật gần
nhất ở 250–360 ft, và không có gì rơi vào khoảng giữa.

Cũng loại bỏ có chủ đích: số hạng bias toàn cục −0.40 ft (giải mã từ chính probe
đó dựa trên public leaderboard — tức fit vào nhiễu của public LB), và bản thân
cell probe.

---

## Cấu trúc repo

```
notebooks/rogii_tvt_pipeline.ipynb   Pipeline đầu-cuối, chạy được trên Kaggle
src/geosteer.py                      Feature builder (cùng code, import được)
experiments/                         Các script đo lường đứng sau những bảng số
docs/FINDINGS.md                     Nhật ký cô đọng những gì đã thử
```

Các script trong `experiments/` cần dữ liệu thi đặt tại `tr/train/`:

| Script | Đo cái gì |
|---|---|
| `cv_ablation.py` | CV GroupKFold và ablation theo họ feature |
| `tracker_comparison.py` | RMSE tracker độc lập và các quy tắc kết hợp |
| `row_thinning.py` | Ảnh hưởng của việc lấy thưa dòng train tới độ chính xác và thời gian fit |
| `reliability_gate.py` | Self-check chỉ trên lateral (kết quả âm) |

## Chạy notebook

Cần GPU cho phần stack CatBoost/LightGBM; tự động chuyển sang CPU nếu không có.

1. `DRY_RUN_WELLS = 120` (mặc định) → chạy trọn trong ~8 phút. Kiểm tra mục 5 in
   ra `SUSPECT: none` và mục 9 cho `corr(dp_err, pf_err)` thấp.
2. Đặt `DRY_RUN_WELLS = 0` rồi **Save & Run All** (~3 giờ).

Kaggle chạy lại toàn bộ notebook khi commit, nên chạy đủ ở chế độ tương tác cộng
thêm một lần commit sẽ tốn gấp đôi quota — hãy kiểm chứng trên subset rồi commit
một lần duy nhất.

Hai lưu ý tinh chỉnh, cả hai đều đã đo:

* **Lấy thưa dòng gần như miễn phí.** Các dòng eval cách nhau 1 ft và tự tương
  quan rất cao. `THIN=1` → 12.097 (fit 141 s); `THIN=3` → **11.913** (73 s);
  `THIN=6` → 12.135 (33 s). Lấy mỗi 3 dòng vừa nhanh hơn *vừa* tốt hơn một chút.
* **16 seed PF là đủ.** 32 seed tốn 3.9 s/giếng so với 2.2 s ở 16 seed, mà không
  có cải thiện đo được nào.

## Phần headroom còn lại nằm ở đâu

Thang oracle 2 nói một đa thức bậc ba theo MD chuẩn hoá đạt 2.3–3.1 ft: quỹ đạo
một giếng chỉ là ~4 con số. Giải pháp này dự đoán TVT theo từng dòng rồi mới
chiếu lên cơ sở đó *sau*. **Dự đoán thẳng các hệ số** — một hồi quy theo giếng
với 4 đầu ra thay vì hồi quy theo dòng với 1 đầu ra — là thí nghiệm hiển nhiên
tiếp theo và chưa từng được thử.

## Ghi nhận

Hai ý tưởng được tham khảo từ các notebook chia sẻ công khai trong cuộc thi:
particle filter đa seed có trọng số likelihood, và phép chiếu đa thức robust cho
`U = TVT + Z`. Các lớp khai thác leak và dò leaderboard của chúng có chủ đích
không được mang sang, vì những lý do đã ghi ở trên.
