# NumPy in 8 Days 八天學會NumPy
 
## Day 1 NumPy 陣列的基本操作
- 安裝與載入 NumPy
- 建立 NumPy array (陣列)
    - numpy.array() 函式
    - 使用 numpy.arange() 與 numpy.linspace() 函式產生等差一維陣列
    - 建立多維陣列
    - numpy.zeros()、numpy.ones()、numpy.empty()
    - 使用隨機函式產生陣列的元素
        - 隨機產生指定形狀的陣列
        - 隨機產生一維陣列的元素
    - 隨機產生不同分佈的陣列元素
- NumPy陣列的索引和切片 (Slicing)
- NumPy 陣列的常用屬性

## Day 2 NumPy 陣列進階操作
- NumPy 陣列重塑
    - flatten() 與 ravel()
    - reshape()
    - resize()
- 軸 (axis) 與維度 (dimension)
    - 一維陣列的軸
    - 二維陣列的軸
    - 三維陣列的軸
    - numpy.newaxis 增加軸數
- NumPy 陣列的合併與分割
    - 合併：numpy.concatenate(), numpy.stack(), numpy.hstack(), numpy.vstack()
    - 分割：numpy.split()、numpy.hsplit()、numpy.vsplit()
- 迭代
- 搜尋與排序
    - 顯示最大值和最小值：amax()、amin()、max()、min()
    - 顯示最大值和最小值的索引：argmax() 與 argmin()
    - 找出符合條件的元素：where
    - nonzero
    - 排序：sort() 與 argsort()

## Day 3 NumPy 陣列運算及數學 Universal Functions (ufunc)
- 四則運算
- sum()
- 次方 numpy.power()
- 平方根 numpy.sqrt()
- 歐拉數 (Euler's number) 及指數函式 np.exp()
- 對數函式
- 取近似值
- 取絕對值：numpy.abs(), numpy.absolute(), numpy.fabs()

## Day 4 NumPy 陣列邏輯函式 (Logic Functions)
- 陣列內容
    - numpy.isnan()
    - numpy.isfinite()
    - numpy.isinf()、numpy.isposinf()、numpy.isneginf()
    - numpy.isnat()
- 陣列型別偵測
    - numpy.isscalar()
    - numpy.isreal()、numpy.iscomplex()、numpy.isrealobj()、numpy.iscomplexobj()
- 比較
    - 比較 2 個陣列是否相同：numpy.array_equal()、numpy.array_equiv()
    - 比較：等於/不等於、大於/大於或等於、小於/小於或等於
- 邏輯操作
- numpy.all()、numpy.any()

## Day 5 NumPy 統計函式 Universal Functions (ufunc)
- 順序統計量 (Order Statistics)
    - 最大值和最小值
        - numpy.maximum(), numpy.minimum()
        - numpy.fmax(), numpy.fmin()
        - numpy.nanmax(), numpy.nanmin()
    - 百分位數：numpy.percentile(), numpy.nanpercentile()
    - 分位數：numpy.quantile(), numpy.nanquantile()
- 平均數與變異數
    - 平均值：mean(), nanmean()
    - 平均值：average()
    - 計算中位數：median(), nanmedian()
    - 計算標準差：std(), nanstd()
    - 計算變異數：var(), nanvar()
- 相關性
    - 相關係數：corrcoef()
    - 互相關 (Cross-correlation)
    - 共變異數：cov()
- Histogram
- digitize()

## Day 6 NumPy I/O
- numpy.save()、numpy.savez()、numpy.load()
- savetxt() 與 loadtxt()
    - numpy.savetxt()
    - numpy.loadtxt()
- genfromtxt()
    - 將文字檔內容讀取並正確分隔Column
    - 選擇要讀取的 Column
    - 缺值處理
    - 資料轉換

## Day 7 NumPy 的矩陣函式與線性代數應用
- 矩陣乘積
    - 點積 (dot product)：numpy.dot(a, b)
    - 內積：numpy.inner()
    - 外積：numpy.outer()
    - 矩陣乘法：numpy.matmul() 與 @
- 矩陣操作
    - 跡：numpy.trace()
    - 行列式 (Determinant)：numpy.linalg.det()
    - 反矩陣：numpy.linalg.inv()
    - 轉置 (Transpose)：numpy.transpose()
    - numpy.linalg.eig()
    - 秩：numpy.linalg.matrix_rank()
    - numpy.linalg.solve()
- 特殊矩陣
    - 單位矩陣：numpy.identity()
    - 單位矩陣：numpy.eye()
    - 單對角陣列 (Diagonal)：numpy.diagonal() 與 numpy.diag()
    - 三角矩陣：numpy.tri()
    - 上三角矩陣 (Upper Triangular)、下三角矩陣 (Lower Triangular)：numpy.triu()、numpy.tril()
- 矩陣分解 (Matrix Decomposition)
    - Cholesky分解：numpy.linalg.cholesky()
    - QR分解：numpy.linalg.qr()
    - SVD分解：numpy.linalg.svd()

## Day 8 NumPy 結構化陣列 (Structured Arrays)
- 資料型別 (dtype)
- 結構化陣列 (Structured Arrays)
- RecordArray：numpy.recarray()
