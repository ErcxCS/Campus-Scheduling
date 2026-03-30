import re
import sys
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"

SCHEDULE_DEFAULTS = {
    ("eng", "B24"): {"final_days": 8, "midterm_days": 10},
    ("eng", "G25"): {"final_days": 9, "midterm_days": 10},
    ("ubf", "B24"): {"final_days": 8, "midterm_days": 10},
}

FACULTIES = ["eng", "ubf"]
SEMESTERS = ["B24", "G25"]


@dataclass
class Config:
    faculty: str
    semester: str
    is_midterm: bool
    is_demo: bool
    timeout: int
    seed: int | None
    num_days: int
    slots_per_day: int
    experiment_id: int | str
    runs_path: Path
    course_path: Path
    room_path: Path
    demo_path: Path
    reference_path: Path | None

    @staticmethod
    def from_args(args) -> "Config":
        faculty = args.faculty
        semester = args.semester
        is_midterm = args.midterm

        # Resolve paths
        faculty_dir = DATA_DIR / faculty
        semester_dir = faculty_dir / semester
        course_path = semester_dir / "courses.xlsx"
        room_path = faculty_dir / "rooms.xlsx"

        exam_type_str = "midterm" if is_midterm else "final"
        demo_path = DATA_DIR / "demo" / f"{faculty}_{semester}_{exam_type_str}.json"

        # Reference schedules (may not exist for all faculty/semester combos)
        exam_ref_str = "midterms" if is_midterm else "finals"
        reference_path = semester_dir / "reference" / exam_ref_str
        if not reference_path.exists():
            reference_path = None

        # Num days: use registry default, allow override
        key = (faculty, semester)
        if args.num_days is not None:
            num_days = args.num_days
        elif key in SCHEDULE_DEFAULTS:
            days_key = "midterm_days" if is_midterm else "final_days"
            num_days = SCHEDULE_DEFAULTS[key][days_key]
        else:
            num_days = 10 if is_midterm else 8

        # Experiment ID
        runs_path = RUNS_DIR
        if args.analyze is not None:
            experiment_id = args.analyze
        elif args.name is not None:
            experiment_id = args.name
        else:
            experiment_id = next_experiment_id(runs_path)

        return Config(
            faculty=faculty,
            semester=semester,
            is_midterm=is_midterm,
            is_demo=args.demo,
            timeout=args.timeout,
            seed=args.seed,
            num_days=num_days,
            slots_per_day=9,
            experiment_id=experiment_id,
            runs_path=runs_path,
            course_path=course_path,
            room_path=room_path,
            demo_path=demo_path,
            reference_path=reference_path,
        )

    def validate(self):
        if not self.course_path.exists():
            sys.exit(f"Error: Course file not found: {self.course_path}")
        if not self.room_path.exists():
            sys.exit(f"Error: Room file not found: {self.room_path}")
        if self.is_demo and not self.demo_path.exists():
            print(f"Warning: Demo file not found: {self.demo_path}")
            print("Will fall back to full optimization.")

    @property
    def dataset_label(self) -> str:
        return f"{self.faculty}_{self.semester}"

    @property
    def has_reference(self) -> bool:
        return self.reference_path is not None

    def exp_dir_name(self) -> str:
        exam_type = "midterm" if self.is_midterm else "final"
        return f"exp{self.experiment_id}_{exam_type}"

    def exp_path(self) -> Path:
        return self.runs_path / self.exp_dir_name()


def next_experiment_id(runs_path: Path) -> int:
    if not runs_path.exists():
        return 0
    nums = []
    for d in runs_path.iterdir():
        if d.is_dir():
            m = re.match(r"exp(\d+)_", d.name)
            if m:
                nums.append(int(m.group(1)))
    return max(nums, default=-1) + 1
