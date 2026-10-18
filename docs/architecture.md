# ארכיטקטורה: MVP לשחזור תמונות

> מבוסס על `docs/spec.md` (סעיפים 5, 7–10) ועל `docs/plan.md`. מסמך תכנון, אין בו קוד.

## 1. עקרונות
- **צינור ליניארי של פונקציות טהורות ככל האפשר** – כל שלב מקבל נתונים ומחזיר תוצאה בלי לגעת בדיסק, כך שקל לבדוק אותו עם תמונות סינתטיות.
- **רק שני מודולים כותבים לדיסק (`ingest`, `storage`) ומודול אחד מתזמר (`pipeline`)** – כך הכלל "המקור לא משתנה" נאכף במקום אחד.
- **ההשוואה עצמאית** – `compare` קורא את המקור ואת התוצאה מהדיסק ולא מקבל דבר מהעיבוד מלבד המסכה, כדי שבאג בעיבוד לא יסתיר את עצמו.
- **ספים וכלים נבחרים ב-`config.yaml`** – החלפת כלי או שינוי סף אינם דורשים שינוי קוד.

## 2. מודולים (`restoration/`)

| מודול | מקבל | מחזיר | הערה |
|---|---|---|---|
| `config` | נתיב `config.yaml` | `Config` (קריאה בלבד) + `config_version` (hash של הקובץ) | הגרסה נכנסת למפתח מניעת הכפילות, כי הגדרות אחרות = תוצאה אחרת. |
| `ingest` | נתיב קלט, `Config` | `IngestResult`: מזהה משימה, hash, נתיב המקור השמור, מידות, ערוצים, DPI אם קיים | בודק תקינות ומעתיק את המקור לקריאה בלבד; קובץ פגום → `IngestError`. |
| `masking` | תמונה, הגדרות מסכה | `DamageMask`: מסכה בינארית + אחוז שטח + מספר אזורים | נפרד מהניקוי, כי גם LaMa צריך מסכה וכך אפשר להחליף כל אחד בנפרד. |
| `clean` | תמונה, `DamageMask`, כלי ניקוי | `StageResult` | מפעיל כלי מסוג inpainter ומוודא שאין שינוי מחוץ למסכה. |
| `sharpen` | תמונה, כלי חידוד | `StageResult` | מפעיל כלי מסוג enhancer; המסכה הרלוונטית היא כל התמונה. |
| `compare` | נתיב מקור, נתיב תוצאה, מסכת נזק, `Config` | `Comparison`: SSIM והפרש מחוץ למסכה, רשימת פנים (מיקום, גודל, SSIM, דגלים) | יישור לפי מידות (כולל הקטנה אם הכלי הגדיל), ומדדים על מקור ותוצאה בלבד. |
| `decide` | `IngestResult`, `Comparison`, שגיאות, `Config` | `Decision`: סטטוס + רשימת סיבות | פונקציה טהורה של כללי סעיף 9, ולכן נבדקת בטבלת מקרים. |
| `report` | `Job` + כל תוצאות השלבים | dict של הדוח (סעיף 5 כאן) + טקסט Markdown | לא כותב לדיסק בעצמו; הדוח נבנה גם למשימה שנכשלה. |
| `storage` | `Job`, תמונות, דוח | נתיבים שנשמרו | מקום יחיד שמכיר את מבנה `storage/` ואת כללי שמות הקבצים. |
| `pipeline` | נתיב קלט, `Config` | `Job` סופי | מתזמר את כל השלבים, תופס חריגות ומבטיח סטטוס סופי ודוח לכל קלט. |
| `restore.py` | ארגומנטים משורת הפקודה | קוד יציאה + הדפסת סטטוס ונתיב הדוח | רק מפרסר ארגומנטים וקורא ל-`pipeline`, כדי שכל הלוגיקה תהיה ניתנת לבדיקה. |

**ייצוג תמונה:** מערך numpy מסוג uint8 בערוצים של המקור (אפור או BGR), כי זה הפורמט הטבעי של OpenCV וכל הכלים המועמדים מקבלים אותו.

## 3. ממשק אחיד לכלי עיבוד (`restoration/tools/`)

כל כלי הוא מחלקה עם אותו חוזה, וכלים נבחרים לפי שם ב-config:

| שדה / פעולה | תיאור |
|---|---|
| `name`, `version` | נכנסים לדוח ("איך בוצע"). |
| `kind` | `inpainter` (חייב מסכה) או `enhancer` (בלי מסכה). |
| `params` | dict ההגדרות מ-config, נשמר בדוח כמו שהוא. |
| `process(ToolInput) → ToolOutput` | הפעולה היחידה. |
| `ToolInput` | תמונה + מסכה (או ריק) + מגבלת זמן. |
| `ToolOutput` | תמונה + `scale` (1 אם לא הוגדל) + מסכת השינוי בפועל + הערות/אזהרות + משך. |

- **שני סוגי כלים במקום ממשק לכל כלי** – LaMa מחליף את OpenCV inpaint ו-Real-ESRGAN מחליף את unsharp mask בלי שינוי ב-`clean`/`sharpen`.
- **`scale` בפלט** – Real-ESRGAN מגדיל את התמונה, וכך `compare` יודע להתאים מידות לפני ההשוואה.
- **רישום כלים במילון אחד (registry) לפי שם** – config בוחר `clean.tool: opencv_inpaint`, והוספת כלי היא קובץ אחד + שורה ברישום.
- **כלים מרוחקים (API בענן) מממשים את אותו חוזה** – החלטת התשתית הפתוחה לא משנה את שאר הקוד.
- **MVP מתחיל עם שני כלים:** `opencv_inpaint` (TELEA/NS כפרמטר) ו-`opencv_unsharp`.

### 3.1 שילוב כלי AI (ניסוי ב', ראה `docs/decisions.md` החלטה 1)

שלושת הכלים המועמדים לא נכנסים בצורה נקייה לשני הסוגים הקיימים, ולכן הממשק מורחב בשלושה שדות בלבד, בלי ממשק נפרד לכל כלי:

| תוספת | למה |
|---|---|
| `kind: restorer` (סוג שלישי) | כלי שמזהה נזק בעצמו ומתקן, בלי לקבל מסכה (Bringing Old Photos עם `--with_scratch`). בניגוד ל-`inpainter` הוא מחזיר את המסכה שיצר. |
| `scope`: `mask` / `faces` / `full_image` | האזור שהכלי רשאי לשנות. OpenCV inpaint = `mask`, CodeFormer = `faces`, Real-ESRGAN ו-Bringing Old Photos = `full_image`. נכנס לדוח ולבדיקות. |
| `requires_gpu`, `license`, `runner` | `runner`: `local` או `remote`. כלי עם `requires_gpu` רץ רק ב-`remote`. הרישיון נרשם בדוח, ובכלי לא מסחרי יש הערה בקובץ הכלי. |

**מיפוי הכלים:**

| כלי | `kind` | `scope` | `scale` | מסכה ב-`ToolOutput` | תוצאה בהשוואה |
|---|---|---|---|---|---|
| `opencv_inpaint` | inpainter | mask | 1 | המסכה שהתקבלה | מחוץ למסכה חייב להיות זהה |
| `bopbtl_scratch` | restorer | full_image | 1 | מסכת השריטות של הכלי | אין ערובה לאי-שינוי מחוץ למסכה, ולכן `compare` בודק ומסמן |
| `codeformer` (`fidelity=0.7`) | enhancer | faces | 1 | תיבות הפנים שזוהו | כל תמונה עם פנים ששונו מקבלת `review` |
| `realesrgan_x4` | enhancer | full_image | 4 | כל התמונה | `compare` מקטין לגודל המקור לפני ההשוואה |

**כללים לשילוב:**
- **הרצה מרוחקת אינה משנה את החוזה.** כלי `remote` מעלה את התמונה, ממתין ומחזיר `ToolOutput` רגיל; כשל או חריגת זמן הם `failed` לפי סעיף 6. ב-Colab אין שירות לקרוא לו, ולכן בניסוי ההרצה ידנית ב-notebook, ובקוד ה-MVP החיבור ל-Colab או לשירות אחר נקבע בהחלטת התשתית.
- **מסכה בפועל היא חובה בפלט.** `compare` משתמש ב-`ToolOutput.mask` ולא באמונה שהכלי נשאר בגבולות המסכה, ולכן כלי `full_image` לא יכול להפר בשקט את הכלל "מחוץ למסכה לא משתנה". הוא יקבל `review` או `failed` לפי הספים.
- **סדר השלבים נשאר.** `restorer` תופס את מקום `masking` + `clean` (הוא עושה את שניהם), ו-`enhancer` תופס את `sharpen`. `pipeline` מדלג על `masking` כשהכלי הנבחר הוא `restorer`.
- **כלי שמשנה פנים (`scope: faces` או `full_image` עם שיפור פנים) מוסיף את הסיבה `face_regenerated` ל-`Decision`.** הסיבה הזו אף פעם לא מניבה `done`.
- **ב-`config.yaml`** נבחר הכלי בשם (`clean.tool: bopbtl_scratch`) והפרמטרים (`fidelity: 0.7`) נשמרים בדוח כמו שהם.
- **תלויות כבדות (PyTorch, ה-weights) לא נכנסות ל-`requirements.txt`.** הן חיות בסביבת ההרצה המרוחקת, והקובץ המקומי של כלי AI הוא עטיפה דקה.

## 4. מודל נתונים: משימה (`Job`)

נשמרת כ-`storage/jobs/<job_id>.json` ומתעדכנת בכל מעבר מצב.

| שדה | תיאור |
|---|---|
| `job_id` | `YYYYMMDD-HHMMSS-<8 תווים מה-hash>` – ממוין לפי זמן וקריא לאדם. |
| `source_name` | שם הקובץ המקורי שהמשתמש העלה. |
| `sha256` | hash של המקור, לזיהוי כפילויות ולאימות שלא השתנה. |
| `config_version` | hash של `config.yaml` בזמן הריצה. |
| `status` | `pending` → `processing` → `done` / `review` / `failed`. |
| `stages` | רשימה: שם שלב, סטטוס, זמן התחלה וסיום, שגיאה. |
| `paths` | `original`, `mask`, `result`, `report_json`, `report_md`. |
| `decision_reasons` | רשימת סיבות לסטטוס (למשל `small_face`, `change_outside_mask`). |
| `error` | הודעת שגיאה אם נכשל. |
| `created_at`, `updated_at` | זמנים ב-ISO 8601. |

- **מניעת כפילות לפי `sha256` + `config_version`** – אותה תמונה עם אותן הגדרות מחזירה את המשימה הקיימת במקום לעבד שוב.
- **חידוש אחרי כשל = הרצה מחדש של כל המשימה באותו `job_id`** – העיבוד דטרמיניסטי ומהיר יחסית, ולכן אין צורך בשמירת שלבי ביניים.
- **מעברי מצב מותרים מוגדרים במקום אחד** – מעבר לא חוקי (למשל `done` → `processing` בלי `--force`) הוא שגיאה.

**מבנה `storage/`:**
```
storage/originals/<job_id>.<ext>                # עותק המקור, לקריאה בלבד
storage/results/<name>_restored_<job_id>.png    # תוצאה (PNG ללא איבוד מידע)
storage/results/<name>_mask_<job_id>.png        # מסכת הנזק
storage/reports/<job_id>.json | .md             # הדוח
storage/jobs/<job_id>.json                      # מצב המשימה
storage/logs/restore.log                        # לוג שגיאות
```
- **התוצאה תמיד PNG** – דחיסת JPEG הייתה מוסיפה שינויים שלא נעשו בכוונה ומשבשת את ההשוואה.

## 5. מבנה הדוח (JSON, לפי סעיף 8)

```json
{
  "schema_version": 1,
  "job_id": "20261018-142233-a1b2c3d4",
  "status": "review",
  "status_reasons": ["small_face"],
  "files": {
    "original": "storage/originals/20261018-142233-a1b2c3d4.jpg",
    "result": "storage/results/grandpa_restored_20261018-142233-a1b2c3d4.png",
    "mask": "storage/results/grandpa_mask_20261018-142233-a1b2c3d4.png",
    "original_sha256": "…"
  },
  "actions": [
    {"action": "inpaint", "reason": "scratches and stains", "mask": "…_mask_….png",
     "area_pct": 2.4, "regions": 17},
    {"action": "sharpen", "reason": "conservative sharpening", "mask": "full_image",
     "area_pct": 100.0}
  ],
  "execution": {
    "config_version": "…",
    "steps": [
      {"stage": "clean", "tool": "opencv_inpaint", "version": "4.10.0",
       "params": {"method": "telea", "radius": 3}, "duration_s": 0.8, "errors": []},
      {"stage": "sharpen", "tool": "opencv_unsharp", "version": "4.10.0",
       "params": {"amount": 0.5, "sigma": 1.0}, "duration_s": 0.1, "errors": []}
    ],
    "total_duration_s": 1.4
  },
  "comparison": {
    "outside_mask_ssim": 0.97,
    "outside_mask_changed_pct": 0.3,
    "faces": [{"box": [120, 80, 40, 40], "size_px": 40, "ssim": 0.93,
               "flags": ["below_min_size"]}]
  },
  "uncertainty": {
    "estimated_fills": "2.4% of image inpainted (estimated content)",
    "limitations": ["face smaller than 64px – fidelity not verifiable"],
    "human_review_recommended": true
  }
}
```

- **`schema_version`** – מאפשר לשנות את הדוח בהמשך בלי לשבור דוחות ישנים.
- **ארבעת חלקי סעיף 8 ממופים ישירות:** מזהה וקישורים → `files`, מה והיכן → `actions`, איך → `execution`, אי-ודאות → `uncertainty`.
- **`comparison` נפרד מ-`actions`** – כך רואים בדוח מה העיבוד טען שעשה ומה ההשוואה העצמאית מצאה, כנדרש ב"הדוח אינו הוכחה".
- **קודי סיבה קבועים באנגלית (`small_face`)** – נוחים לבדיקות ולסינון, וה-Markdown מתרגם אותם לטקסט קריא.

## 6. זרימת נתונים: מ-`restore.py` ועד הדוח

```
restore.py <image>
  │  פרסור ארגומנטים
  ▼
pipeline.run(path, config)
  1. config.load            → Config + config_version
  2. ingest                 → IngestResult         (קובץ פגום → failed)
     storage: כפילות?       → אם קיימת: מחזיר את המשימה הקיימת ועוצר
     storage: שמירת מקור לקריאה בלבד; Job = pending → processing
  3. masking(image)         → DamageMask
  4. clean(image, mask)     → StageResult           (כלי inpainter מ-config)
  5. sharpen(image)         → StageResult           (כלי enhancer מ-config)
     storage: שמירת תוצאה ומסכה כ-PNG
  6. compare(original_path, result_path, mask) → Comparison   (קריאה מחדש מהדיסק)
  7. decide(...)            → Decision (done / review / failed + סיבות)
  8. report(job, ...)       → JSON + Markdown
     storage: שמירת הדוח; Job = status סופי
  ▼
restore.py: הדפסת סטטוס + נתיב הדוח; קוד יציאה 0 (done), 2 (review), 1 (failed)
```

- **כל חריגה בשלבים 3–8 נתפסת ב-`pipeline`** – המשימה מסומנת `failed`, הדוח נכתב עם מה שנאסף, והמקור נשאר כפי שהוא.
- **חריגה ממגבלת זמן מטופלת כשגיאת כלי** – לפי סעיף 9 זו "עצירת טיפול", ולכן הסטטוס הוא `failed`.
- **בסוף הריצה נבדק שוב ה-hash של המקור** – אם השתנה, זו שגיאה קריטית שמסמנת `failed`, כי זה הכלל החשוב ביותר בפרויקט.
- **קודי יציאה שונים לכל סטטוס** – סקריפט עתידי יוכל לסנן תמונות שדורשות בדיקה בלי לקרוא את הדוח.

## 7. קשר לתוכנית העבודה
- שלבים 1–3 (ניסוי) נשארים סקריפטים עצמאיים ב-`experiments/`; בשלב 6 הלוגיקה עוברת ל-`tools/opencv_inpaint` ול-`masking`.
- שלב 4 → `ingest`, `storage`; שלב 5 → `Job`, `config`; שלבים 6–7 → `masking`, `clean`, `sharpen`, `tools/`; שלבים 8–9 → `compare`; שלב 10 → `decide`; שלב 11 → `report`; שלב 12 → `pipeline`, `restore.py`.

## 8. שאלות שהארכיטקטורה משאירה פתוחות
- שיטת בדיקת הפנים (SSIM או embeddings) – משפיעה רק על `compare`.
- האם להגדיל רזולוציה – נתמך דרך `scale`, אך ברירת המחדל 1.
- טיפול בסריקה צבעונית של תמונה שחור-לבן – משפיע על `ingest` בלבד.
