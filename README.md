# Campus Exam Scheduling System

An automated constraint programming solution for university exam timetabling. This system uses Google OR-Tools to generate conflict-free schedules for Midterm and Final exams, optimizing for room capacity, student overlap, and instructor availability.

## Installation

0. **Get the Code (Demo Branch)**:
   ```bash
   git clone https://github.com/ErcxCS/Campus-Scheduling.git
   cd Campus-Scheduling
   git checkout demo
   ```

1. **Prerequisites**: Ensure you have Python 3.8+ installed.
2. **Create a Virtual Environment** (Recommended):
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```


## How to Run

The system is controlled via the command line:

```bash
python -m src.main [arguments]
```

### Arguments

| Argument | Description |
|---|---|
| *(No Argument)* | Runs a standard Final Exam optimization for Engineering faculty, B24 semester (8 days, 600s timeout). |
| `--faculty {eng,ubf}` | Select faculty (default: `eng`). |
| `--semester {B24,G25}` | Select semester dataset (default: `B24`). |
| `--midterm` | Switches the mode to Midterm Exams. Default is Finals. |
| `--demo` | Demo Mode: Uses a pre-calculated solution ("warm start") to finish in seconds. |
| `--timeout N` | Sets the solver time limit in seconds (default: 600). |
| `--seed N` | Sets a random seed for reproducibility. |
| `--num-days N` | Override schedule duration in days. |
| `--analyze ID` | Skips the solver and runs the analysis suite on a specific Experiment ID. |
| `--name NAME` | Custom experiment name (default: auto-increment). |

## Important Note on Analysis

The Analysis Module (graphs, heatmaps, and comparison reports) requires manual reference schedules for comparison. Analysis runs automatically when reference data exists for the selected faculty/semester combination.

Currently, reference data is available for: **Engineering faculty, B24 semester** (finals and midterms).

## Usage Examples

### 1. Standard Final Exam Run (Engineering, B24)

Runs optimization for 10 minutes (default) and generates a schedule.

```bash
python -m src.main
```

### 2. Final Demo

Uses cached hints to generate a valid schedule in ~10-30 seconds.

```bash
python -m src.main --demo
```

### 3. Midterm Demo

```bash
python -m src.main --midterm --demo
```

### 4. UBF Faculty, B24 Semester

```bash
python -m src.main --faculty ubf
```

### 5. Engineering, G25 Semester

```bash
python -m src.main --semester G25
```

### 6. Re-Analyze an Old Experiment

Regenerates the graphs and reports for experiment #3 without running the solver again.

```bash
python -m src.main --analyze 3
```

## Project Structure

```
Campus-Scheduling/
├── src/                    # Python source code
│   ├── main.py             # CLI entry point and orchestration
│   ├── config.py           # Config dataclass, path resolution, defaults
│   ├── solver.py           # OR-Tools CP-SAT constraint logic
│   ├── entities.py         # Data models (Course, Room, Department, TimeSlot)
│   ├── reports.py          # Excel schedule generation and formatting
│   ├── analytics.py        # Statistical analysis and plotting
│   └── utils.py            # File I/O and helper functions
├── data/
│   ├── eng/                # Engineering faculty
│   │   ├── rooms.xlsx      # Room data (shared across semesters)
│   │   ├── B24/            # Semester data
│   │   │   ├── courses.xlsx
│   │   │   └── reference/  # Manual schedules for comparison
│   │   └── G25/
│   │       └── courses.xlsx
│   ├── ubf/                # UBF faculty (same structure)
│   ├── demo/               # Pre-computed warm-start solutions
│   └── timetables/         # Midterm off-time timetable
├── runs/                   # Experiment outputs
├── requirements.txt
└── README.md
```
