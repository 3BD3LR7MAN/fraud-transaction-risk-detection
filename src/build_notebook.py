from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / 'notebooks' / 'Fraud_Risk_Detection.ipynb'
out.parent.mkdir(parents=True, exist_ok=True)
nb = nbf.v4.new_notebook()
nb.metadata = {
    'colab': {'name': 'Fraud_Risk_Detection.ipynb', 'provenance': []},
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'}
}
nb.cells = [
    nbf.v4.new_markdown_cell("""# Fraud / Transaction Risk Detection

دفتر تشغيل لمشروع ترتيب المعاملات المشبوهة **للمراجعة البشرية** باستخدام بيانات ULB/Worldline الحقيقية المنشورة على OpenML. لا يصدر النموذج قراراً آلياً بالاحتيال ولا يرفض معاملة.

**للتشغيل:** فك ضغط ملفات المشروع في Google Drive داخل `MyDrive/Fraud_Transaction_Risk_Detection` ثم عدّل المسار في الخلية التالية إذا اختلف. أول تشغيل ينزّل ملف البيانات (حوالي 150 MB) من مرآة عامة."""),
    nbf.v4.new_code_cell("""# 1) ربط Google Drive وتحديد مجلد المشروع
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
PROJECT_DIR = Path('/content/drive/MyDrive/Fraud_Transaction_Risk_Detection')
print('Project:', PROJECT_DIR)
assert (PROJECT_DIR / 'src' / 'train_fraud.py').exists(), 'استخرج ملفات المشروع إلى المجلد المحدد أولاً.'"""),
    nbf.v4.new_code_cell("""# 2) تثبيت الاعتمادات
%pip -q install pandas numpy scikit-learn scipy matplotlib seaborn joblib requests"""),
    nbf.v4.new_markdown_cell("""## البيانات ومشكلة عدم التوازن

مجموعة OpenML 42175 تضم معاملات بطاقات موسومة؛ الاحتيال فئة نادرة جداً. نقارن النماذج باستخدام PR-AUC وPrecision وRecall، وليس Accuracy فقط. نحافظ على ترتيب `Time` ونقسّم البيانات زمنياً لتقليل تسرب المستقبل إلى التدريب."""),
    nbf.v4.new_code_cell("""# 3) تنزيل البيانات الخام مرة واحدة
import subprocess, sys
DATA_PATH = PROJECT_DIR / 'data' / 'raw' / 'creditcard.csv'
if not DATA_PATH.exists():
    subprocess.run([sys.executable, str(PROJECT_DIR / 'src' / 'download_data.py'), '--output', str(DATA_PATH)], check=True)
print('Dataset file:', DATA_PATH, '| MB:', round(DATA_PATH.stat().st_size/1e6, 1))"""),
    nbf.v4.new_code_cell("""# 4) نظرة عامة سريعة على CSV
import sys
sys.path.insert(0, str(PROJECT_DIR / 'src'))
from train_fraud import read_dataset
import pandas as pd

df = read_dataset(DATA_PATH)
print('Shape:', df.shape)
print('Fraud counts:')
display(df['Class'].value_counts().rename(index={0:'Legitimate', 1:'Fraud'}))
print('Fraud rate:', f"{df['Class'].mean():.3%}")
display(df[['Time', 'Amount', 'Class']].describe())"""),
    nbf.v4.new_markdown_cell("""## Baseline ومحسّن، ثم عتبة مراجعة

يشغّل السكربت Logistic Regression مرجعياً وRandom Forest موزوناً للفئة النادرة. يختار PR-AUC الأعلى على validation. بعد ذلك يختار عتبة تحقق Recall مستهدفاً قدره 70% على validation، ويقيس الاختبار الزمني مرة واحدة. ملف `threshold_tradeoffs.csv` يقارن أحجام المراجعة والإيجابيات الكاذبة والاحتيال الفائت عند عتبات متعددة."""),
    nbf.v4.new_code_cell("""# 5) تشغيل كامل التدريب والتقييم وإنتاج الملفات
subprocess.run([
    sys.executable, str(PROJECT_DIR / 'src' / 'train_fraud.py'),
    '--data', str(DATA_PATH), '--output', str(PROJECT_DIR), '--target-recall', '0.70'
], check=True)"""),
    nbf.v4.new_code_cell("""# 6) قراءة المقاييس والتقرير
import json
metrics = json.loads((PROJECT_DIR / 'reports' / 'metrics.json').read_text())
display(pd.Series(metrics))
print((PROJECT_DIR / 'reports' / 'Evaluation_Report_EN.md').read_text())"""),
    nbf.v4.new_code_cell("""# 7) عرض الرسوم الناتجة
from IPython.display import Image, display
for name in ['eda_imbalance_amount.png', 'fraud_rate_over_time.png', 'test_confusion_matrix.png', 'test_precision_recall_curve.png', 'permutation_importance.png']:
    p = PROJECT_DIR / 'reports' / 'figures' / name
    print(name)
    display(Image(filename=str(p)))"""),
    nbf.v4.new_markdown_cell("""## ملاحظة أعمال وسلامة

اختيار العتبة هنا مثال منهجي، لا توصية مالية نهائية. تكلفة مراجعة كل تنبيه، والخسارة غير المستردة عند تفويت احتيال، وسعة المحللين لم تُعطَ؛ يجب أن يحددها مالك العمل قبل التشغيل. كل تنبيه يذهب للمراجعة البشرية فقط. خصائص V1–V28 مكونات مجهولة لا تحمل تفسيراً تجارياً مباشراً.""")
]
nbf.write(nb, out)
print(out)
