# 🌿 Plant Disease Advisor — the app

**Action-Group Conformal Prediction for Safe Plant Disease Advisory under Lab-to-Field Shift**

A farmer uploads a photo of a leaf. The app does **not** just say "this is disease X". It asks
*"Can I be sure which **treatment** is needed?"*

* sure about one treatment group → it gives that group's advice
* unsure between treatments → **"ask an expert (KVK)"**
* the photo looks unlike anything it was trained on → **"photo not recognised"**

---

## 1. The two parts of the project

| Part | Where it runs | What it does |
|---|---|---|
| `train_model_colab.ipynb` (separate file) | **Google Colab** (free GPU) | downloads the data, trains, calibrates, gives you `trained_model.zip` |
| this folder `plant_disease_advisor/` | **your computer** (no GPU, no PyTorch) | loads the trained model and shows the web app |

```
Google Colab                                   your computer
──────────────────────────                     ──────────────────────────────────────────────
train_model_colab.ipynb  ── trained_model.zip ──►  saved_model/plant_model.onnx    (the network)
                                                   saved_model/model_info.json     (classes, thresholds)
                                                              │ loaded by
                                                              ▼
                                                   src/predictor.py ◄── app/streamlit_app.py  (web page)
                                                                    ◄── app/api.py            (REST API)
                                                   src/database.py  ──► database/predictions.db
```

## 2. Folder structure

```
plant_disease_advisor/
├── README.md              this file
├── config.py              all paths and settings in one place
├── requirements.txt       packages for the app (no PyTorch)
│
├── app/
│   ├── streamlit_app.py   the web page (frontend)
│   └── api.py             optional REST API (backend, e.g. for a mobile app)
│
├── src/
│   ├── disease_info.py    38 classes → 6 treatment groups
│   ├── image_utils.py     photo → numbers (resize 256, crop 224, normalise)
│   ├── conformal.py       group scores, prediction set, decision
│   ├── predictor.py       loads saved_model/ and predicts ONE photo
│   ├── advice.py          reads data/advice.csv
│   └── database.py        SQLite: predictions + feedback
│
├── data/
│   └── advice.csv         what to tell the farmer for each group
├── saved_model/           ← put the files from trained_model.zip here
├── database/              created automatically (prediction history)
└── tests/                 pytest tests
```

`saved_model/` already contains a **demo model** (MobileNetV3, only the last layer trained, on a CPU), so the app runs
straight away. Replace it with your Colab model.

## 3. Step 1 — Train on Google Colab

1. Open https://colab.research.google.com → **File → Upload notebook** → `train_model_colab.ipynb`.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. First check: in Step 2 set `QUICK_TEST = True`, then **Runtime → Run all** (~10 min).
4. Real model: set `QUICK_TEST = False`, **Runtime → Run all** again (~1–1.5 h, keep the tab open).
5. At the end `trained_model.zip` downloads to your computer.

## 4. Step 2 — Put the model into the app (Windows PowerShell)

```powershell
cd C:\projects\plant_disease_advisor
Remove-Item -Recurse -Force .\saved_model
Expand-Archive "$HOME\Downloads\trained_model.zip" -DestinationPath .\saved_model
dir .\saved_model          # must show plant_model.onnx and model_info.json
```

## 5. Step 3 — Run the app (Windows PowerShell)

First time only:
```powershell
cd C:\projects\plant_disease_advisor
python -m venv myenv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Every time:
```powershell
cd C:\projects\plant_disease_advisor
.\myenv\Scripts\Activate.ps1
streamlit run app/streamlit_app.py
```
The browser opens http://localhost:8501. The sidebar shows which model is loaded and its scores.

Optional REST API (second PowerShell window, same venv): `uvicorn app.api:app --reload` → open http://localhost:8000/docs

Tests: `pytest -q` → `10 passed`

## 6. The method in short

| | Formula | Where |
|---|---|---|
| Probabilities | p = softmax(z / T) | `conformal.softmax` |
| Group score (ours) | s(x, g) = 1 − Σ<sub>y∈g</sub> p(y \| x) | `conformal.group_scores` |
| Prediction set | C(x) = {g : s(x, g) ≤ q̂<sub>g</sub>}  (q̂<sub>g</sub> learned on Colab, Step 18) | `conformal.prediction_sets` |
| Decision | \|C\| = 1 → advise · \|C\| ≥ 2 → refer · else unknown | `conformal.decide` |
| Unknown photo | E(x) = −T · log Σ exp(z/T) > limit | `conformal.energy_score` |

The temperature T, the thresholds q̂<sub>g</sub> and the energy limit are in `saved_model/model_info.json`.
All results of the training run (for the report / paper) are in `saved_model/results.json` and `results.png`.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `No saved model found in saved_model` | unzip `trained_model.zip` into `saved_model\` (step 2) |
| `The saved model's classes do not match` | the model was not made by `train_model_colab.ipynb` |
| `ModuleNotFoundError: No module named 'src'` | run the command from the project folder (the one with `config.py`) |
| `'streamlit' is not recognized` | activate the venv: `.\.venv\Scripts\Activate.ps1` |
| `running scripts is disabled on this system` | once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `WinError 4551` (a DLL is blocked) | Windows Smart App Control: move the project out of `Downloads` (e.g. `C:\projects`), or run the app in your Linux VM |
| Colab: training is very slow | the GPU is not on: Runtime → Change runtime type → T4 GPU |

## 8. Before publication

* Fill the `source` column of `data/advice.csv` (marked "TO CITE") with ICAR / TNAU references.
* Ask a plant-pathology faculty member to check the disease → group table (`src/disease_info.py` and Step 3 of the notebook).
