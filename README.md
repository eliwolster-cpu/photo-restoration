# שחזור תמונות היסטוריות

כלי שורת פקודה (Python) שמשחזר באופן שמרני תמונות היסטוריות סרוקות, תמונה אחת בכל פעם: מנקה נזק מקומי (סדקים, כתמים, שריטות), מחדד בזהירות, משווה את התוצאה למקור באופן עצמאי, ושומר תוצאה, דוח קצר וסטטוס (`done` / `review` / `failed`).

**כללי יסוד:** קובץ המקור לעולם לא משתנה (נשמר לקריאה בלבד ומאומת ב-hash). נאמנות למקור חשובה מיופי: משלימים פרטים רק בתוך מסכת הנזק, ופיקסלים מחוץ למסכה נשארים ללא שינוי. במקרה של ספק הסטטוס הוא `review`.

## מה כבר בנוי
המפרט המלא ב-[docs/spec.md](docs/spec.md), והתוכנית ב-[docs/plan.md](docs/plan.md).

| שלב | מה | קובץ |
|---|---|---|
| 1 | קו בסיס של inpainting ב-OpenCV (TELEA ו-NS) עם מסכה ידנית, ותמונת השוואה צד-לצד | `experiments/baseline_inpaint.py` |
| 2 | מסכת נזק אוטומטית פשוטה (top-hat / black-hat) וחישוב IoU מול מסכה ידנית | `experiments/auto_mask.py` |
| 4 | שלד הפרויקט וקליטת קובץ: בדיקת תקינות, hash, מזהה משימה, שמירת המקור לקריאה בלבד וזיהוי כפילויות | `restoration/ingest.py`, `restore.py` |

בנוסף: `experiments/make_mask.py` (הופך סימון אדום על עותק התמונה למסכה שחור-לבן), ובדיקות pytest לכל הנ"ל, כולל מקרי קצה (PNG עם שקיפות, 16 ביט, שמות בעברית, קבצים פגומים, "פצצות דחיסה").

**עדיין לא בנוי:** שלב 3 (ניסוי על 5 תמונות) ושלבים 5 ואילך. כרגע `restore.py` רק קולט את הקובץ, ואין עדיין ניקוי, חידוד, השוואה או דוח.

## התקנה והרצה
נדרש Python 3.11.

```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

pytest                                   # כל הבדיקות
python restore.py photo.jpg              # קליטת תמונה (שומר עותק לקריאה בלבד ב-storage/originals)

# ניסויים
python experiments/baseline_inpaint.py photo.jpg mask.png --out experiments/output
python experiments/auto_mask.py photo.jpg --manual mask.png
```

## מבנה התיקיות
```
docs/          # spec.md, plan.md, architecture.md
experiments/   # סקריפטי ניסוי ההיתכנות (תמונות ופלטים אינם נשמרים ב-git)
restoration/   # מודולי הצינור (כרגע: ingest)
tests/         # pytest ותמונות סינתטיות קטנות
storage/       # לא ב-git: originals/ (קריאה בלבד), results/, reports/, jobs/
restore.py     # נקודת כניסה (CLI)
```

- [docs/spec.md](docs/spec.md) – אפיון
- [docs/plan.md](docs/plan.md) – תוכנית עבודה לפי שלבים
- [docs/architecture.md](docs/architecture.md) – ארכיטקטורת ה-MVP
