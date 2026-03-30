# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Constraint programming solution for university exam timetabling using Google OR-Tools CP-SAT solver. Generates conflict-free midterm and final exam schedules optimizing for room capacity, student overlap, and time distribution.

Two key domain axes: **Faculty** (eng = Engineering, ubf = UBF) and **Semester** (B24 = Bahar/Spring 2024, G25 = Güz/Fall 2025). Each faculty has its own course and room data; semesters represent time periods applicable to any faculty.

## Commands

```bash
# Setup
python -m venv venv && .\venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run scheduling (always from project root)
python -m src.main                                    # eng, B24, Final, 600s
python -m src.main --faculty ubf --semester B24       # UBF faculty
python -m src.main --semester G25                     # eng, G25
python -m src.main --midterm --demo                   # Midterm demo mode
python -m src.main --timeout 300 --seed 42            # Custom timeout + seed
python -m src.main --analyze 3                        # Re-run analysis on experiment #3
python -m src.main --name my_experiment               # Custom experiment name
```

There are no tests in this project.

## Architecture

**Data flow**: `main.py` (CLI + Config) -> `entities.py` (load data) -> `solver.py` (CP-SAT) -> `reports.py` (Excel output) -> `analytics.py` (plots)

All source code lives in `src/` and runs as `python -m src.main`.

### Key modules

- **`src/config.py`** - `Config` dataclass, `SCHEDULE_DEFAULTS` registry (num_days per faculty/semester), path resolution via `pathlib.Path`, input validation. All paths derived from `PROJECT_ROOT`.
- **`src/entities.py`** - Data models with class-level state: `Course.course_list`, `Room.room_list`, `TimeSlot.slot_list` populated via class methods. These act as global registries.
- **`src/solver.py`** - Core constraint logic. Accepts a `Config` object. Defines interval variables, seating constraints, department/year overlap prevention, lab/regular room assignment, daily balance, and off-time enforcement. Entry point: `exam_scheduling_main(cfg)`.
- **`src/reports.py`** - Generates Excel outputs: room-time timetable, per-department schedules, faculty-wide schedule.
- **`src/analytics.py`** - Concurrency heatmaps, room sharing histograms, distribution plots, automated-vs-manual comparison. Requires manual reference schedules in `data/{faculty}/reference/`. Entry point: `analysis(cfg)`.
- **`src/utils.py`** - File I/O helpers. `read_fac_xlsxs()` loads both automated output and manual reference for comparison.

### Data organization

```
data/
├── eng/                        # Engineering faculty
│   ├── rooms.xlsx              # Room data (shared across semesters)
│   ├── B24/                    # Semester folder
│   │   ├── courses.xlsx
│   │   └── reference/          # Manual schedules for comparison
│   │       ├── finals/
│   │       └── midterms/
│   └── G25/
│       └── courses.xlsx
├── ubf/                        # UBF faculty (same structure)
│   ├── rooms.xlsx
│   └── B24/
│       └── courses.xlsx
├── demo/                       # Pre-computed warm-start JSON files
│   └── {faculty}_{semester}_{exam_type}.json
└── timetables/                 # Midterm off-time data
```

### Output structure

Each solver run creates `runs/exp{N}_{final|midterm}/` containing: `exam_schedule.xlsx`, `beautified_exam_schedule.xlsx`, `faculty_schedule.xlsx`, `department_schedules/`, and `plots/`.

### Analysis

Analysis requires manual reference schedules for the selected faculty/semester combo. Currently available for: eng/B24 (finals and midterms). The system auto-detects whether reference data exists via `Config.has_reference`.

## Key constraints in solver

1. Exam duration fits within 2-slot intervals
2. All students seated across assigned rooms (cumulative capacity)
3. No room over-capacity at any time slot
4. Same department + year courses cannot overlap
5. Mandatory courses in same department cannot overlap
6. Lab courses assigned to lab rooms only
7. Midterm year splitting: years 1&3 in first half, years 2&4 in second half
8. Daily exam balance per department (hard constraint with tolerance)
9. Off-time slots enforced per day
10. Each exam assigned to exactly one faculty building
