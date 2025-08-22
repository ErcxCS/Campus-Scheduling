import numpy as np
import pandas as pd
import random
from ortools.sat.python import cp_model
from matplotlib import pyplot as plt
import os
from openpyxl import load_workbook
from openpyxl.styles import PatternFill


class Department:
    departments = []
    department_names = []
    department_ids = []

    def __init__(self, department_name: str):
        self.courses = []
        self.curriculums = {i: [] for i in range(1, 5)}
        self.id = len(Department.departments)
        Department.department_ids.append(self.id)

        self.name = department_name
        words = self.name.split(" ")
        self.short = "".join(word[0] for word in words)
        Department.department_names.append(department_name)
        Department.departments.append(self)

    def add_course(self, course):
        if course not in self.courses:
            self.courses.append(course)
            self.curriculums.setdefault(course.year, []).append(course)

    def get_course(self, course_code: str):
        for course in self.courses:
            if course_code == course.course_code:
                return course
        return None

    @classmethod
    def get_dep(cls, dep_short: str):
        for dep in Department.departments:
            if dep_short == dep.short:
                return dep
        return None

    @classmethod
    def get_department(cls, department_name: str):
        if department_name in cls.department_names:
            idx = cls.department_names.index(department_name)
            return cls.departments[idx]
        return cls(department_name)


class Course:
    course_list = []
    course_codes = set()
    _by_code_and_inst = dict()
    pass_course_midterm = ["FİZ 176", "GIDA 324", "EEM 370"]

    def __init__(
        self,
        id: int,
        department: Department,
        course_name: str,
        year: int,
        n_students: int,
        course_code: str,
        instructor_id: int,
        requires_lab: bool,
        mandatory: str
    ):
        self.id = int(id)
        self.n_students = int(n_students)
        self.departments = [department]
        self.course_code = course_code
        self.instructor_id = instructor_id
        self.requires_lab = requires_lab
        self.year = year
        self.course_name = course_name
        self.mandatory = mandatory

        self.dep_short = ' '.join(dep.short for dep in self.departments)

        # Each course is assumed to have exactly one block of duration 2
        self.blocks = [2]
        
    def get_duration(self):
        return self.blocks[0]
    
    def get_dep_shorts(self):
        return ' '.join(dep.short for dep in self.departments)

    @staticmethod
    def read_courses(path: str, is_midterm: bool):
        df = pd.read_excel(path, index_col=None, header=0)
        for i, row in enumerate(df.values):
            (
                department_name,
                course_code,
                course_name,
                year,
                mandatory,
                instructor_id,
                n_students,
                requires_lab
            ) = row

            if is_midterm and course_code in Course.pass_course_midterm:
                continue

            dep = Department.get_department(department_name)
            inst_id = int(instructor_id)
            key = (course_code, inst_id)

            if key in Course._by_code_and_inst:
                existing = Course._by_code_and_inst[key]
                existing.departments.append(dep)
                existing.n_students += int(n_students)
                dep.add_course(existing)
            else:
                new_course = Course(
                    id=i,
                    department=dep,
                    course_name=course_name,
                    year=int(year),
                    n_students=int(n_students),
                    course_code=course_code,
                    instructor_id=inst_id,
                    requires_lab=(requires_lab == 1),
                    mandatory=mandatory
                )
                dep.add_course(new_course)
                Course.course_codes.add(course_code)
                Course._by_code_and_inst[key] = new_course
                Course.course_list.append(new_course)


class TimeSlot:
    slot_list: list = []
    ids: np.ndarray
    offs: np.ndarray

    def __init__(self, id: int, is_off: bool):
        self.id = int(id)
        self.is_off = bool(is_off)

    @staticmethod
    def generate_week(n_days: int, n_slots_per_day: int, off_slot_lists_per_day):
        total_slots = n_days * n_slots_per_day
        TimeSlot.ids = np.arange(total_slots)
        TimeSlot.offs = np.zeros(total_slots, dtype=bool)

        for day in range(n_days):
            off_slots = off_slot_lists_per_day[day]
            for s in off_slots:
                global_idx = day * n_slots_per_day + s
                TimeSlot.offs[global_idx] = True

        TimeSlot.slot_list = [
            TimeSlot(id=idx, is_off=off)
            for idx, off in zip(TimeSlot.ids, TimeSlot.offs)
        ]


class Room:
    rooms: pd.DataFrame
    room_list: list = []
    ids: np.ndarray
    capacities: np.ndarray
    off_times_dict: dict = None

    def __init__(self, id: int, room_code: str, capacity: int, c_type: str, off_times: list[int] = None):
        self.id = int(id)
        self.capacity = int(capacity)
        self.room_code = room_code
        self.is_lab = (c_type == "Lab")
        self.off_times = off_times

    @staticmethod
    def read_classroom_data(path: str, num_days: int, slots_per_day: int, is_midterm: bool):
        df = pd.read_excel(path)
        df = df[df["Room"].notna()][["Room", "Capacity", "Type"]]

        room_ids = []
        room_codes = []
        room_caps = []

        labs = []
        regulars = []
        if is_midterm:
            off_timetable = midterm_timetable(num_days, slots_per_day)
            Room.off_times_dict = off_timetable

        for i, row in enumerate(df.values):
            room_code, capacity, c_type = row
            room_ids.append(i)
            room_caps.append(int(capacity))
            room_codes.append(room_code)

            if is_midterm:
                room_obj = Room(i, room_code, int(capacity), c_type, off_timetable[room_code])
            else:
                room_obj = Room(i, room_code, int(capacity), c_type, None)

            if room_obj.is_lab:
                labs.append(room_obj)
            else:
                regulars.append(room_obj)

        # Always put regular rooms first, then labs
        Room.room_list = regulars + labs
        Room.ids = np.array(room_ids)
        Room.capacities = np.array(room_caps)
        Room.rooms = pd.DataFrame({
            "id": room_ids,
            "room_codes": room_codes,
            "capacities": room_caps
        })

    @staticmethod
    def find_by_code(room_code: str):
        for room in Room.room_list:
            if room.room_code == room_code:
                return room
        raise Exception(f"room not found: {room_code} is not in room_list")


def get_off_chunks(slot_list):
    """
    From a list of TimeSlot objects (with .id and .is_off),
    return a list of (start, end) for each consecutive off‐chunk.
    """
    off_chunks = []
    current = []

    for s in sorted(slot_list, key=lambda x: x.id):
        if s.is_off:
            current.append(s.id)
        else:
            if current:
                off_chunks.append((current[0], current[-1] + 1))
                current = []
    if current:
        off_chunks.append((current[0], current[-1] + 1))
    return off_chunks


def build_timetable2(courses, rooms, horizon, n_days, solver, start_vars, in_room_vars, seat_vars, exp_path):
    """
    Builds a per‐day timetable DataFrame, writes it to "exam_schedule.xlsx",
    and returns a list of (day_name, DataFrame) tuples.
    """
    import datetime

    room_ids = [r.id for r in rooms]
    timetable = pd.DataFrame("", index=range(horizon), columns=room_ids)

    for e in courses:
        s_val = solver.Value(start_vars[e.id])
        dur = e.get_duration()
        for t in range(s_val, s_val + dur):
            for r in rooms:
                if solver.Value(in_room_vars[(e.id, r.id)]) == 1:
                    used_seats = solver.Value(seat_vars[(e.id, r.id)])
                    info = f"{e.dep_short} {e.course_code}({used_seats}:{e.year})"
                    if timetable.at[t, r.id]:
                        timetable.at[t, r.id] += "|" + info
                    else:
                        timetable.at[t, r.id] = info

    time_indices = [
        datetime.time(h + 8, 30).strftime("%H:%M") for h in range(9)
    ]
    day_length = horizon // n_days
    all_days = []
    day_names = [f"Day-{i}" for i in range(1, n_days + 1)]

    for d in range(n_days):
        start_row = d * day_length
        end_row = (d + 1) * day_length
        df_day = timetable.iloc[start_row:end_row, :].copy()
        df_day.index = time_indices
        df_day.columns = [
            f"{rooms[i].room_code}({rooms[i].capacity if rooms[i].is_lab else rooms[i].capacity // 2})"
            for i in range(len(rooms))
        ]
        df_day["Day"] = day_names[d]
        all_days.append((day_names[d], df_day))

    combined = pd.concat([df for _, df in all_days])
    raw_schedule_path = os.path.join(exp_path, "exam_schedule.xlsx")
    combined.to_excel(raw_schedule_path, index=False)
    return all_days


def department_exam_schedule(departments, courses, rooms, horizon, n_days, solver, start_vars, in_room_vars, exp_path):
    day_length = horizon // n_days
    import datetime
    timetables_path = os.path.join(exp_path, "department_schedules")
    os.makedirs(timetables_path, exist_ok=True)
    for dep in departments:
        data = []
        for year, year_courses in dep.curriculums.items():
            for course in year_courses:
                start_var = solver.Value(start_vars[course.id])
                nth_day = start_var // day_length + 1
                start_time = start_var % day_length
                date_time = datetime.time(start_time + 8, 30).strftime("%H:%M")
                day_str = f"day - {nth_day}"

                assigned_rooms = ""
                for r in rooms:
                    if solver.Value(in_room_vars[(course.id, r.id)]) == 1:
                        assigned_rooms += r.room_code + ","
                assigned_rooms = assigned_rooms[:-1]
                data.append({
                    'Year': year,
                    'Course ID': course.course_code,
                    # 'Course Name': course.course_name,
                    'Day': nth_day,
                    'Starting Slot': start_time,
                    'Rooms': assigned_rooms
                })
        dep_df = pd.DataFrame(data)
        excel_name = dep.name + ".xlsx"
        doc_path = os.path.join(timetables_path, excel_name)
        dep_df.to_excel(doc_path, index=False)


def faculty_exam_schedule(courses, rooms, horizon, n_days, solver, start_vars, in_room_vars, exp_path):
    day_length = horizon // n_days
    import datetime
    data = []
    for course in courses:
        start_var = solver.Value(start_vars[course.id])
        nth_day = start_var // day_length + 1
        start_time = start_var % day_length
        date_time = datetime.time(start_time + 8, 30).strftime("%H:%M")
        day_str = f"day - {nth_day}"

        assigned_rooms = ""
        total_room_cap = 0
        for r in rooms:
            if solver.Value(in_room_vars[(course.id, r.id)]) == 1:
                assigned_rooms += r.room_code + ","
                total_room_cap += r.capacity if r.is_lab else r.capacity // 2

        assigned_rooms = assigned_rooms[:-1]
        num_rooms_used = len(assigned_rooms.split(","))

        deps = course.get_dep_shorts()
        deps = deps.split(" ")
        deps = ",".join(deps)
        data.append({
            'Departments': deps,
            'Year': course.year,
            'Course ID': course.course_code,
            # 'Course Name': course.course_name,
            'Day': nth_day,
            'Starting Slot': start_time,
            'Assigned Rooms': assigned_rooms,
            'Num Rooms Used': num_rooms_used,
            'Total Room Cap': total_room_cap,
            "Num Students": course.n_students
        })
    fac_df = pd.DataFrame(data)
    excel_name = "faculty_schedule" + ".xlsx"
    doc_path = os.path.join(exp_path, excel_name)
    fac_df.to_excel(doc_path, index=False)


def plot_exam_per_day(exams_per_day, num_days):
    x = list(range(1, num_days + 1))
    y = [exams_per_day.get(day - 1, 0) for day in x]

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, marker="o", label="Total Exams")
    plt.xlabel("Day")
    plt.ylabel("Number of Exams")
    plt.title("Exams Scheduled per Day")
    plt.legend()
    plt.xticks(x)
    plt.show()


def plot_dep_year_exam_counts(dep_year_per_day, num_days):
    keys = dep_year_per_day.keys()
    dep_set = sorted({dep_short for (_, dep_short, _) in keys})

    fig, axes = plt.subplots(len(dep_set), 1, figsize=(12, 3 * len(dep_set)), sharex=True)
    years = [1, 2, 3, 4]
    year_colors = ["blue", "orange", "green", "purple"]

    for idx, dep_short in enumerate(dep_set):
        ax = axes[idx]
        for year in years:
            y_vals = [
                dep_year_per_day.get((day, dep_short, year), 0)
                for day in range(num_days)
            ]
            nonzero = [(day, count) for day, count in zip(range(num_days), y_vals) if count != 0]
            if nonzero:
                days_f, vals_f = zip(*nonzero)
                ax.plot(days_f, vals_f, marker="o", color=year_colors[year - 1], label=f"Year {year}")

        ax.set_title(dep_short)
        ax.set_ylabel("Exam Count")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    plt.xlabel("Day")
    plt.xticks(range(num_days), [f"Day {d}" for d in range(num_days)])
    plt.suptitle("Exam Distribution by Department and Year per Day", y=1.02)
    plt.tight_layout()
    plt.show()


def mission_report(solver, start_vars, slots_per_day, in_room_vars, num_days):
    """
    1. Counts and prints how many exams per day.
    2. Prints department-year breakdown per day.
    3. Displays two plots (exam-per-day and dept-year-per-day).
    """
    # 1) Count total exams per day
    exams_per_day = {}
    for e in Course.course_list:
        s_val = solver.Value(start_vars[e.id])
        day_idx = s_val // slots_per_day
        exams_per_day[day_idx] = exams_per_day.get(day_idx, 0) + 1

    print("Exams scheduled per day:")
    total_exams = 0
    for day, count in sorted(exams_per_day.items()):
        print(f"  Day {day}: {count} exams")
        total_exams += count
    plot_exam_per_day(exams_per_day, num_days)
    print(f"Total exams: {total_exams}\n")

    # 2) Department/Year per-day breakdown
    dep_year_per_day = {}
    for e in Course.course_list:
        s_val = solver.Value(start_vars[e.id])
        day_idx = s_val // slots_per_day
        key = (day_idx, e.departments[0].short, e.year)
        dep_year_per_day[key] = dep_year_per_day.get(key, 0) + 1

    print("Department-Year exam count per day:")
    for (day, dep_short, year) in sorted(dep_year_per_day):
        cnt = dep_year_per_day[(day, dep_short, year)]
        print(f"  Day {day}, {dep_short}, Year {year}: {cnt} exams")
    plot_dep_year_exam_counts(dep_year_per_day, num_days)


def exam_scheduling_main(experiment_no: int, is_midterm: bool, num_days: int, slots_per_day: int):
    # ---------------------------
    # 0) Parameters & Data Loading
    # ---------------------------
    # TODO!: Fix exam start times
    # TODO: There should be no exam starting in a room while there is an active exam going on
    # within active time slot chunk

    course_xlsx = "./data/BerkData2.xlsx"
    room_xlsx = "./data/New Microsoft Excel Worksheet.xlsx"

    Course.read_courses(course_xlsx, is_midterm)
    Room.read_classroom_data(room_xlsx, num_days, slots_per_day, is_midterm)

    # Off-time per day (e.g. lunch slot = 4)
    off_by_day = [[4] for _ in range(num_days)]
    off_by_day[4] = off_by_day[4] + [5]  # Day 5 has two off slots
    if num_days >= 10:
        off_by_day[9] = off_by_day[9] + [5]

    #off_by_day[3] = off_by_day[3] + [5, 6]  # simulations of 5i exams

    TimeSlot.generate_week(num_days, slots_per_day, off_by_day)
    horizon = num_days * slots_per_day

    # ---------------------------
    # 1) Build CP Model
    # ---------------------------
    model = cp_model.CpModel()

    # (1) Interval / start & end variables for each exam
    start = {}
    end = {}
    interval = {}
    if is_midterm:
        for e in Course.course_list:
            dur = e.get_duration()
            if e.year in {1, 3}:
                start[e.id] = model.NewIntVar(0, horizon // 2 - dur, f"start_e{e.id}")
                end[e.id] = model.NewIntVar(0, horizon // 2, f"end_e{e.id}")
            else:
                start[e.id] = model.NewIntVar(horizon // 2, horizon - dur, f"start_e{e.id}")
                end[e.id] = model.NewIntVar(horizon // 2, horizon, f"end_e{e.id}")

            model.Add(end[e.id] == start[e.id] + dur)
            interval[e.id] = model.NewIntervalVar(start[e.id], dur, end[e.id], f"interval_e{e.id}")
    else:
        for e in Course.course_list:
            dur = e.get_duration()
            start[e.id] = model.NewIntVar(0, horizon - dur, f"start_e{e.id}")
            end[e.id] = model.NewIntVar(0, horizon, f"end_e{e.id}")
            model.Add(end[e.id] == start[e.id] + dur)
            interval[e.id] = model.NewIntervalVar(start[e.id], dur, end[e.id], f"interval_e{e.id}")

    # (2) Seat & in_room variables
    seat = {}
    in_room = {}
    for e in Course.course_list:
        for r in Room.room_list:
            cap = r.capacity // 2 if not r.is_lab else r.capacity
            seat[(e.id, r.id)] = model.NewIntVar(0, cap, f"seat_e{e.id}_r{r.id}")
            in_room[(e.id, r.id)] = model.NewBoolVar(f"in_room_e{e.id}_r{r.id}")
            model.Add(seat[(e.id, r.id)] <= cap * in_room[(e.id, r.id)])

    # (3) Full-seat constraint
    for e in Course.course_list:
        model.Add(sum(seat[(e.id, r.id)] for r in Room.room_list) == e.n_students)

    # (4) Optional intervals + no-overlap per room
    opt_int_per_room = {r.id: [] for r in Room.room_list}
    for e in Course.course_list:
        for r in Room.room_list:
            opt = model.NewOptionalIntervalVar(
                start[e.id],
                e.get_duration(),
                end[e.id],
                in_room[(e.id, r.id)],
                f"opt_e{e.id}_r{r.id}"
            )
            opt_int_per_room[r.id].append(opt)

    for r in Room.room_list:
        demands = [seat[(e.id, r.id)] for e in Course.course_list]
        cap = r.capacity if r.is_lab else r.capacity // 2
        model.AddCumulative(intervals=opt_int_per_room[r.id], demands=demands, capacity=cap)

        # Also enforce a "no more than 3 simultaneous exams" hard cap per room:
        model.AddCumulative(intervals=opt_int_per_room[r.id],
                            demands=[1] * len(Course.course_list),
                            capacity=3)

    # (X) Exams in the same room that overlap must start at the same time
    """ for r in Room.room_list:
        for i in range(len(Course.course_list)):
            for j in range(i+1, len(Course.course_list)):
                e = Course.course_list[i]
                f = Course.course_list[j]
                dur_e = e.get_duration()
                dur_f = f.get_duration()

                # 1) b1: e finishes ≤ f starts
                b1 = model.NewBoolVar(f"e{e.id}_before_f{f.id}")
                model.Add(start[e.id] + dur_e <= start[f.id]).OnlyEnforceIf(b1)
                model.Add(start[e.id] + dur_e > start[f.id]).OnlyEnforceIf(b1.Not())

                # 2) b2: f finishes ≤ e starts
                b2 = model.NewBoolVar(f"f{f.id}_before_e{e.id}")
                model.Add(start[f.id] + dur_f <= start[e.id]).OnlyEnforceIf(b2)
                model.Add(start[f.id] + dur_f > start[e.id]).OnlyEnforceIf(b2.Not())

                # 3) overlap ⇔ not (e before f or f before e)
                overlap = model.NewBoolVar(f"overlap_e{e.id}_f{f.id}_r{r.id}")
                model.AddBoolAnd([b1.Not(), b2.Not()]).OnlyEnforceIf(overlap)
                model.AddBoolOr([b1, b2]).OnlyEnforceIf(overlap.Not())

                # 4) if both in room r AND they overlap, force same start
                model.Add(start[e.id] == start[f.id]).OnlyEnforceIf([
                    in_room[(e.id, r.id)],
                    in_room[(f.id, r.id)],
                    overlap
                ]) """

    # (5) Dept/Year no-overlap
    dept_year_intervals = {}
    for c in Course.course_list:
        for dep in c.departments:
            key = (dep.id, c.year)
            dept_year_intervals.setdefault(key, []).append(interval[c.id])

    for intervals in dept_year_intervals.values():
        model.AddNoOverlap(intervals)

    # (6) Lab vs non-lab enforcement
    for e in Course.course_list:
        if e.requires_lab:
            for r in Room.room_list:
                if not r.is_lab:
                    model.Add(seat[(e.id, r.id)] == 0)
        else:
            for r in Room.room_list:
                if r.is_lab:
                    model.Add(seat[(e.id, r.id)] == 0)

    # (7) Link "in_room ⇒ seat ≥ 1"
    for e in Course.course_list:
        for r in Room.room_list:
            model.Add(seat[(e.id, r.id)] >= 1).OnlyEnforceIf(in_room[(e.id, r.id)])

    # (8) Day-within-slot constraints
    for e in Course.course_list:
        day_e = model.NewIntVar(0, num_days - 1, f"day_e{e.id}")
        model.AddDivisionEquality(day_e, start[e.id], slots_per_day)
        model.Add(start[e.id] >= day_e * slots_per_day)
        model.Add(start[e.id] + e.get_duration() <= (day_e + 1) * slots_per_day)

    # (9) Off-times: no exam may overlap any off-chunk
    off_chunks = get_off_chunks(TimeSlot.slot_list)
    for e in Course.course_list:
        day_e = model.NewIntVar(0, num_days - 1, f"day_off_e{e.id}")
        model.AddDivisionEquality(day_e, start[e.id], slots_per_day)
        local_time = model.NewIntVar(0, slots_per_day - 1, f"local_e{e.id}")
        model.Add(local_time == start[e.id] - day_e * slots_per_day)

        for (o_start, o_end) in off_chunks:
            b1 = model.NewBoolVar(f"b1_e{e.id}_{o_start}_{o_end}")
            b2 = model.NewBoolVar(f"b2_e{e.id}_{o_start}_{o_end}")

            model.Add(end[e.id] <= o_start).OnlyEnforceIf(b1)
            model.Add(start[e.id] >= o_end).OnlyEnforceIf(b2)
            model.AddBoolOr([b1, b2])

    # (10) Balanced exam-per-department constraints
    # spread out each department-year’s exams roughly evenly across the 8 days
    local_day = {}
    # TODO: change the week_len to num_days == 5 for midterm
    # TODO!: this constraint will have issues for midterm scheduling
    # because of 16 exam of department MM?
    if is_midterm:
        week_len = num_days // 2
        for e in Course.course_list:
            local_day[e.id] = model.NewIntVar(0, week_len - 1, f"local_day_e{e.id}")
            if e.year in {1, 3}:
                model.AddDivisionEquality(local_day[e.id], start[e.id], slots_per_day)
            else:
                dur = e.get_duration()
                shifted_start = model.NewIntVar(0, horizon - slots_per_day * week_len - dur,
                                                f"shifted_start_{e.id}")
                model.Add(shifted_start == start[e.id] - slots_per_day * week_len)
                model.AddDivisionEquality(local_day[e.id], shifted_start, slots_per_day)
    else:
        week_len = num_days
        for e in Course.course_list:
            local_day[e.id] = model.NewIntVar(0, week_len - 1, f"local_day_e{e.id}")
            model.AddDivisionEquality(local_day[e.id], start[e.id], slots_per_day)

    count_vars = {}
    for dep in Department.departments:
        for year, exams in dep.curriculums.items():
            for d in range(week_len):
                count_vars[(dep.id, year, d)] = model.NewIntVar(0, len(exams),
                    f"count_dep{dep.id}_yr{year}_d{d}")
                indicators = []
                for e in exams:
                    ind = model.NewBoolVar(f"ind_e{e.id}_d{d}")
                    model.Add(local_day[e.id] == d).OnlyEnforceIf(ind)
                    model.Add(local_day[e.id] != d).OnlyEnforceIf(ind.Not())
                    indicators.append(ind)
                model.Add(count_vars[(dep.id, year, d)] == sum(indicators))

    tolerance = 1
    for dep in Department.departments:
        for year, exams in dep.curriculums.items():
            total = len(exams)
            avg = total / week_len
            low = int(avg)
            high = int(avg) + tolerance
            for d in range(week_len):
                model.Add(count_vars[(dep.id, year, d)] >= low)
                model.Add(count_vars[(dep.id, year, d)] <= high)

    # (11) "Mission active" helper -> minimize how many room‐times are actually used
    # (Optional: we include it in the final objective)
    mission_active = {}
    for r in Room.room_list:
        for t in TimeSlot.slot_list:
            var = model.NewBoolVar(f"mission_active_r{r.id}_t{t.id}")
            mission_active[(r.id, t.id)] = var

            if r.off_times and t.id in r.off_times:
                model.Add(var == 0)

            active_list = []
            for e in Course.course_list:
                b_start = model.NewBoolVar(f"bstart_e{e.id}_{r.id}_before_{t.id}")
                b_end = model.NewBoolVar(f"bend_e{e.id}_{r.id}_after_{t.id}")

                model.Add(start[e.id] <= t.id).OnlyEnforceIf(b_start)
                model.Add(start[e.id] > t.id).OnlyEnforceIf(b_start.Not())

                model.Add(end[e.id] > t.id).OnlyEnforceIf(b_end)
                model.Add(end[e.id] <= t.id).OnlyEnforceIf(b_end.Not())

                active = model.NewBoolVar(f"active_e{e.id}_r{r.id}_t{t.id}")
                model.AddBoolAnd([in_room[(e.id, r.id)], b_start, b_end]) \
                    .OnlyEnforceIf(active)
                model.AddBoolOr([
                    in_room[(e.id, r.id)].Not(),
                    b_start.Not(),
                    b_end.Not()
                ]).OnlyEnforceIf(active.Not())

                active_list.append(active)

            model.AddMaxEquality(var, active_list)

    total_missions = sum(mission_active.values())


    # (12) Final objective: minimize (# of rooms used) + 2 × (room‐times actually used)
    room_usage = []
    for e in Course.course_list:
        for r in Room.room_list:
            room_usage.append(in_room[(e.id, r.id)])
    model.Minimize(sum(room_usage) + 2 * total_missions)

    # ---------------------------
    # 2) Solve & Report
    # ---------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 1200
    solver.parameters.num_search_workers = 12 # 12, 16
    solver.parameters.log_search_progress = True
    print(f"symmetry: {solver.parameters.symmetry_level}")
    solver.parameters.symmetry_level = 3 # 3, 2

    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        run_result = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
        print("Solution status:", run_result)
        # Build the Excel‐output timetable
        exam_type = "_midterm" if is_midterm else "_final"
        exp_path = os.path.join(runs_path, "exp" + str(experiment) + exam_type)
        os.makedirs(exp_path, exist_ok=True)
        build_timetable2(Course.course_list, Room.room_list, horizon, num_days,
                         solver, start, in_room, seat, exp_path)

        department_exam_schedule(Department.departments,
                                 Course.course_list,
                                 Room.room_list,
                                 horizon,
                                 num_days,
                                 solver,
                                 start,
                                 in_room,
                                 exp_path)

        faculty_exam_schedule(
            Course.course_list,
            Room.room_list,
            horizon,
            num_days,
            solver,
            start,
            in_room,
            exp_path
        )
        # Print and plot mission report
        mission_report(solver, start, slots_per_day, in_room, num_days)
        excelify(Department.departments, exp_path, Room.room_list, output_filename="exam_schedule.xlsx")
    else:
        print("No solution found (status {}).".format(status))


def excelify(dep_list: list, exp_path, room_list, output_filename="exam_schedule.xlsx"):
    import pandas as pd

    # Read the combined timetable DataFrame from the Excel file.
    schedule_path = os.path.join(exp_path, output_filename)
    combined_df = pd.read_excel(schedule_path)

    # Ensure "Day" is the first column.
    cols = combined_df.columns.tolist()
    if "Day" in cols and cols[0] != "Day":
        cols.remove("Day")
        cols = ["Day"] + cols
        combined_df = combined_df[cols]

    # Write the DataFrame to Excel using XlsxWriter.
    beautified_path = os.path.join(exp_path, "beautified_" + output_filename)
    writer = pd.ExcelWriter(beautified_path, engine="xlsxwriter")
    combined_df.to_excel(writer, index=False, sheet_name="Schedule")

    workbook = writer.book
    worksheet = writer.sheets["Schedule"]

    # Set column widths for all columns except Day column.
    num_columns = len(combined_df.columns)
    # Set columns 1 to end (i.e., room columns) to width 16 (~110px).
    worksheet.set_column(1, num_columns - 1, 16)
    # Set Day column (column 0) to a width of 12.
    worksheet.set_column(0, 0, 12)

    # Now, merge the Day cells per day group and apply alternate row coloring.
    header_rows = 1  # Header occupies row 0.
    current_day = None
    start_idx = None   # index in combined_df for start of a day group
    group_index = 0    # counts day groups for alternate coloring

    # Define a format for alternate day rows (faint yellow) with center
    # (You can remove 'align' and 'valign' if you want no alignment.)
    yellow_format = workbook.add_format({
        'bg_color': '#FFFFE0', 'align': 'center', 'valign': 'vcenter'
        })

    for i, row in combined_df.iterrows():
        day = row["Day"]
        if day != current_day:
            if current_day is not None:
                first_row = header_rows + start_idx
                last_row = header_rows + i - 1
                worksheet.merge_range(first_row, 0, last_row, 0, current_day)
                if group_index % 2 == 0:
                    for r_idx in range(first_row, last_row + 1):
                        worksheet.set_row(r_idx, None, yellow_format)
                group_index += 1
            current_day = day
            start_idx = i
    if current_day is not None:
        first_row = header_rows + start_idx
        last_row = header_rows + i
        worksheet.merge_range(first_row, 0, last_row, 0, current_day)
        if group_index % 2 == 0:
            for r_idx in range(first_row, last_row + 1):
                worksheet.set_row(r_idx, None, yellow_format)

    # --------------------------
    # Department coloring setup:
    # --------------------------
    # Define a list of colors for departments.
    color_list = [
        "#A9A9A9",
        "#800080",
        "#FFA500",
        "#FFFF00",
        "#FF0000",
        "#008000",
        "#ADD8E6"
    ]
    # Build a mapping from department short to a color.
    dept_colors = {}
    for i, dep in enumerate(dep_list):
        dept_colors[dep.short] = color_list[i % len(color_list)]

    # Create a format for each department (only background color, no centering)
    dept_formats = {}
    for dep_short, color in dept_colors.items():
        dept_formats[dep_short] = workbook.add_format({'bg_color': color})

    # Define a default format (if department info is not found).
    default_format = workbook.add_format({})

    # Initialize a dictionary to count the total number of colored cells
    colored_counts = {dep_short: 0 for dep_short in dept_colors.keys()}

    # -----------------------------------------------
    # Merge cells vertically for identical exam intervals
    # in each room column and apply department colors.
    # We assume each non-empty cell in a room column is in the format:
    #    "{dep_short} {course_code}({n_students})"
    for col in range(1, num_columns):
        row = header_rows  # Start after header.
        while row < header_rows + len(combined_df):
            current_val = combined_df.iloc[row - header_rows, col]
            # Check if current_val is NaN and convert to empty string.
            if pd.isna(current_val):
                current_val = ""
            if current_val != "":
                start_merge = row
                next_row = row + 1
                # Find how many consecutive rows have the same value.
                while (next_row < header_rows + len(combined_df) and
                        combined_df.iloc[next_row - header_rows, col] == current_val):
                    next_row += 1
                end_merge = next_row - 1  # Last row with the same value.
                merged_length = end_merge - start_merge + 1

                # Extract the department short.
                # We assume the cell is like "BM CS101(45)" so split on space.
                try:
                    dep_short = current_val.split()[0]
                    cell_format = dept_formats.get(dep_short, default_format)
                except Exception:
                    cell_format = default_format
                    dep_short = None

                # If we got a valid department short ,
                # and cell_format is not default update the count.
                if dep_short is not None and cell_format != default_format:
                    colored_counts[dep_short] += merged_length

                if end_merge > start_merge:
                    worksheet.merge_range(
                        start_merge,
                        col,
                        end_merge,
                        col,
                        current_val,
                        cell_format
                    )
                else:
                    worksheet.write(start_merge, col, current_val, cell_format)
                row = next_row  # Move pointer past this merged block.
            else:
                row += 1

    writer._save()
    color_offtimes_black(beautified_path)

    # Print out the total number of colored cells for each department.
    print("Colored cells count by department:")
    total_mission = 0
    for dep_short, count in colored_counts.items():
        true_count = count // 2
        total_mission += true_count
        print(f"Department {dep_short}: {true_count} cells")

    print(f"Total mission count: {total_mission}")


def find_black_cells(file_path, sheet_name, slots_per_day, week_offset, off_timetable):
    # Load the workbook and select the worksheet
    wb = load_workbook(file_path)
    ws = wb[sheet_name]

    for i, col in enumerate(ws.iter_cols()):
        room_code = ""
        for j, cell in enumerate(col):
            cell_color = cell.fill.start_color.index

            if j == 0:
                room_code = cell.value.split('\n')[0]
                if room_code not in off_timetable.keys():
                    off_timetable[room_code] = []
                    continue

            if cell_color == 1:
                time = j - 1
                ts = time - (time // slots_per_day) + week_offset
                off_timetable[room_code].append(ts)

    return off_timetable


def midterm_timetable(slots_per_day: int = 9, num_days: int = 10):
    midterm_tb_path = "./data/bahar_midterm.xlsx"
    horizon = slots_per_day * num_days
    off_timetable = {}
    off_timetable = find_black_cells(midterm_tb_path,
                                     'first',
                                     slots_per_day,
                                     0,
                                     off_timetable)
    off_timetable = find_black_cells(midterm_tb_path,
                                     'second',
                                     slots_per_day,
                                     horizon // 2,
                                     off_timetable)
    return off_timetable


def color_offtimes_black(xlsx_path: str):
    if Room.off_times_dict is None:
        return

    wb = load_workbook(xlsx_path)
    ws = wb["Schedule"]

    black_fill = PatternFill(
        bgColor="000000",
        fill_type="solid"
    )

    for i, col in enumerate(ws.iter_cols()):
        room_code = ""
        off_list = None
 
        for j, cell in enumerate(col):
            if cell.value is not None:
                value = str(cell.value)
                room_code = value.split('(')[0]

                if room_code in Room.off_times_dict.keys():
                    off_list = Room.off_times_dict[room_code]
                    continue
                
            if off_list and (j - 1) in off_list:
                cell.fill = black_fill

    wb.save(xlsx_path)


def total_mission_count(df: pd.DataFrame, verbose=0):
    day_mission_count = {}
    day_exam_count = {}

    list_day = df["Day"].unique().tolist()
    for day in list_day:
        day_df = df[df["Day"] == day]
        list_slot = day_df["Starting Slot"].unique().tolist()
        day_mission_count[day] = 0
        day_exam_count[day] = len(day_df)
        # print(f"{day_df['Departments'].value_counts()}, day{day}")
        for slot in list_slot:
            slot_df = day_df[day_df["Starting Slot"] == slot]
            rooms = slot_df["Assigned Rooms"].values.tolist()

            room_codes = ",".join(rooms)
            rooms = set(room_codes.split(","))
            unique_room_count = len(rooms)
            if verbose == 1:
                print(f"Day: {day} - Slot: {slot} - Rooms: {rooms}")
                print(f"unique rooms: {rooms}, count: {unique_room_count}")

            day_mission_count[day] += unique_room_count
    day_mission_count = dict(sorted(day_mission_count.items()))
    mission_count = sum(day_mission_count.values())
    return mission_count, day_mission_count


def read_dfs(experiment_no: int, exam: str = None):
    runs_path = "./runs"
    runs = os.listdir(runs_path)
    run = [r for r in runs if "exp" + str(experiment_no) in r]
    run = run[0]
    exp_path = os.path.join(runs_path, run)
    schedules_folder = "department_schedules"
    if exam:
        schedules_path = "./data/department_schedules_" + exam
    else:
        schedules_path = os.path.join(exp_path, schedules_folder)
    
    department_schedules = os.listdir(schedules_path)
    if "modified" in department_schedules:
        department_schedules.remove("modified")
    if "faculty_schedule_manuel.xlsx" in department_schedules:
        department_schedules.remove("faculty_schedule_manuel.xlsx")

    dep_dfs: dict[str: pd.DataFrame] = {}
    for dep in department_schedules:
        dep_word = dep.split(" ")
        dep_code = dep_word[0][0] + dep_word[1][0]
        xlsx_path = os.path.join(schedules_path, dep)
        df = pd.read_excel(xlsx_path, index_col=None, header=0)
        dep_dfs[dep_code] = df

    
    faculty_xlsx = "faculty_schedule.xlsx"
    faculty_path = os.path.join(exp_path, faculty_xlsx)
    fac_df = pd.read_excel(faculty_path, index_col=None, header=0)

    return dep_dfs, fac_df


def save_to(dep: Department, df: pd.DataFrame, exam: str):
    dfs_paths = "./data/department_schedules_" + exam
    modified_dfs = os.path.join(dfs_paths, "modified")
    os.makedirs(modified_dfs, exist_ok=True)
    xlsx_path = os.path.join(modified_dfs, dep.name + ".xlsx")
    df.to_excel(xlsx_path, index=False)


def add_info(df: pd.DataFrame, dep: Department):
    name_list = []
    course_codes = df["Course ID"].tolist()
    for course_code in course_codes:
        try:
            course = dep.get_course(course_code)
            name_list.append(course.course_name)
        except:
            print(course_code)
    df["Course Name"] = name_list
    return df


def unified_manuel_fac_schedule(dfs: dict[str: pd.DataFrame], course_list: list[Course]):
    data = []
    for course in course_list:
        used_rooms: set[Room] = set()
        date_info: list[int] = list()
        for dep in course.departments:
            dep_df = dfs[dep.short]
            try:
                course_entry = dep_df[dep_df["Course ID"] == course.course_code]
                rooms_used = course_entry["Rooms Used"].values[0]
            except Exception as e:
                print(f"Error: unable to read course with course_code {course.course_code} - \n\t {e}")
                return
            
            used_room_codes = rooms_used.split(",")
            for used_room in used_room_codes:
                try:
                    used_rooms.add(Room.find_by_code(used_room.strip()))
                except Exception as e:
                    print(f"Error retreiving room with room code: {used_room, dep.short} - \n\t {e}")
                    return

            date = course_entry["Date"].values[0]
            date_info.append(date)
        if len(set(date_info)) != 1:
            raise Exception("Different date info")

        
        deps = course.get_dep_shorts()
        deps = deps.split(" ")
        deps = ",".join(deps)
        data.append({
            'Departments': deps,
            'Year': course.year,
            'Course ID': course.course_code,
            # 'Course Name': course.course_name,
            'Day': date,
            'Assigned Rooms': [room.room_code for room in used_rooms],
            'Num Rooms Used': len(used_rooms),
            'Total Room Cap': sum([room.capacity if room.is_lab else room.capacity // 2 for room in used_rooms]),
            "Num Students": course.n_students
        })
    fac_manuel_df = pd.DataFrame(data)
    return fac_manuel_df


def save_manuel_fac(df, exam):
    df_path = "./data/department_schedules_" + exam
    fac_xslx_path = os.path.join(df_path, "faculty_schedule_manuel_" + exam + ".xlsx")
    df.to_excel(fac_xslx_path, index=False)

def read_fac_xlsxs(experiment_no: int, exam: str = None):
    # Construct the path directly using the experiment number
    exp_path = f"./runs/exp{experiment_no}_{exam[:-1]}"
    
    faculty_xlsx = "faculty_schedule.xlsx"
    faculty_path = os.path.join(exp_path, faculty_xlsx)
    fac_df_out = pd.read_excel(faculty_path, index_col=None, header=0)

    df_path = f"./data/department_schedules_{exam}"
    faculty_path = os.path.join(df_path, f"faculty_schedule_manuel_{exam}.xlsx")
    fac_df_manuel = pd.read_excel(faculty_path, index_col=None, header=0)
    
    return fac_df_manuel, fac_df_out

def find_unique_rooms(df: pd.DataFrame) -> list[Room]:
    """
    Finds unique Room objects from the 'Assigned Rooms' column of a DataFrame.

    This version is more efficient and idiomatic pandas, avoiding explicit loops
    by using a combination of string operations and set manipulation directly.
    """
    # Create a list of all room codes by splitting the strings in the Series
    # and then flattening the resulting list of lists.
    all_rooms_list = df['Assigned Rooms'].str.split(',').explode().to_list()
    
    # Use a set comprehension for a concise way to collect unique room objects.
    # The set automatically handles uniqueness.
    unique_rooms = {Room.find_by_code(code) for code in all_rooms_list}

    return list(unique_rooms)

def room_utilization(df: pd.DataFrame, is_manuel: bool = False):
    """
    Calculates room utilization metrics by day and overall.
    
    This version uses pandas' groupby for more efficient and readable
    processing, eliminating the need for nested explicit loops.
    """
    def calculate_total_capacity(df):
        rooms = find_unique_rooms(df)
        return sum(room.capacity if room.is_lab else room.capacity // 2 for room in rooms)
    
    def count_unique_rooms(df):
        return len(df['Assigned Rooms'].str.split(',').explode().unique())
        
    def find_flawed_entries_by_day(df):
        flawed_entries = df[df['Num Students'] > df['Total Room Cap']]
        return flawed_entries.groupby('Day').size()

    def calculate_excess_students_by_day(df):
        # Filter for entries where student count exceeds capacity
        excess_entries = df[df['Num Students'] > df['Total Room Cap']]
        # Calculate the excess students for each flawed entry
        excess_students = excess_entries['Num Students'] - excess_entries['Total Room Cap']
        # Group the excess students by day and sum them
        return excess_students.groupby(excess_entries['Day']).sum()

    if not is_manuel:
        slot_groups = df.groupby(['Day', 'Starting Slot'])
        slot_capacities = slot_groups.apply(calculate_total_capacity)
        slot_student_numbers = slot_groups['Num Students'].sum()
        slot_unique_rooms = slot_groups.apply(count_unique_rooms)
        daily_capacities = slot_capacities.groupby('Day').sum()
        daily_student_numbers = slot_student_numbers.groupby('Day').sum()
        unique_rooms_per_day = df.groupby('Day').apply(count_unique_rooms)
        daily_active_room_time_slots = slot_unique_rooms.groupby('Day').sum()
    else:
        daily_capacities = df.groupby('Day')['Total Room Cap'].sum()
        daily_student_numbers = df.groupby('Day')['Num Students'].sum()
        unique_rooms_per_day = df.groupby('Day')['Num Rooms Used'].sum()
        daily_active_room_time_slots = None
        
        flawed_entries_per_day = find_flawed_entries_by_day(df)
        print(f"Total number of infeasbile assignments: {sum(dict(flawed_entries_per_day).values())}")
        
        # Calculate and print the daily sum of excess students
        daily_excess_students = calculate_excess_students_by_day(df)
        print(f"Daily Sum of Excess Students: {dict(daily_excess_students)}")
        # daily_student_numbers = daily_student_numbers - daily_excess_students * 2

    daily_utilizations = daily_student_numbers / daily_capacities
    overall_utilization = daily_student_numbers.sum() / daily_capacities.sum()

    print(f"Daily Capacities: {list(daily_capacities)}")
    print(f"Daily Student Numbers: {list(daily_student_numbers)}")
    print(f"Overall Utilization: {overall_utilization}")
    print(f"Daily Utilizations: {list(daily_utilizations)}")
    
    if is_manuel:
        active_slots = sum(unique_rooms_per_day)
        print(f"Daily Active Room-Time Slots: {list(unique_rooms_per_day)}")
        print(f"Total Active Room-Time Slots: {sum(unique_rooms_per_day)}")
    else:
        active_slots = sum(daily_active_room_time_slots)
        print(f"Daily Active Room-Time Slots: {list(daily_active_room_time_slots)}")
        print(f"Total Active Room-Time Slots: {active_slots}")
        
    return active_slots, overall_utilization

# Assume the necessary classes (Room, Course, TimeSlot) and other functions 
# (read_fac_xlsxs, find_unique_rooms, room_utilization) are defined as you provided.

def analyze_exam_distribution(df: pd.DataFrame, is_midterm: bool):
    """
    Analyzes the distribution of exams for each department-year tuple,
    correctly handling courses shared between multiple departments.

    Args:
        df: DataFrame containing the schedule.
        is_midterm: Boolean indicating if the schedule is for midterms.

    Returns:
        A dictionary with (department, year) tuples as keys and the
        standard deviation of their daily exam counts as values.
    """
    # 1. Preprocess the DataFrame to handle shared departments
    # Create a new DataFrame where the 'Department' column is split by commas
    # and then "exploded" into separate rows for each department.
    df_processed = df.assign(Departments=df['Departments'].str.split(',')).explode('Departments')

    # 2. Group by the now-separated departments and year
    grouped = df_processed.groupby(['Departments', 'Year'])
    
    distribution_std_dev = {}

    for (department, year), group in grouped:
        # Count exams per day for the current group
        daily_counts = group.groupby('Day').size()
        
        # Determine the set of relevant days for the calculation
        if is_midterm:
            # [cite_start]For midterms, the schedule is split into two halves [cite: 165, 166]
            if year in [1, 3]:
                relevant_days = range(6)  # First half (Days 0-4)
            else: # years 2, 4
                relevant_days = range(6, 11) # Second half (Days 5-9)
        else:
            # For finals, use all available days in the schedule
            num_days = df['Day'].max()
            relevant_days = range(1, num_days +1)
        
        # Ensure all relevant days are included in the series (with 0 if no exams)
        # This prevents the standard deviation from being skewed by missing days.
        daily_counts = daily_counts.reindex(list(relevant_days), fill_value=0)
            
        # 3. Calculate the standard deviation for the balanced daily counts
        distribution_std_dev[(department, year)] = np.std(daily_counts)
        
    return distribution_std_dev


def get_daily_exam_counts(df: pd.DataFrame, is_midterm: bool):
    """
    Calculates the daily exam counts for each department-year tuple.

    Args:
        df: DataFrame containing the schedule.
        is_midterm: Boolean indicating if the schedule is for midterms.

    Returns:
        A dictionary where keys are (department, year) tuples and values 
        are lists of exam counts for each relevant day.
    """
    # Preprocess to correctly handle shared departments
    df_processed = df.assign(Departments=df['Departments'].str.split(',')).explode('Departments')
    
    grouped = df_processed.groupby(['Departments', 'Year'])
    
    daily_counts_dict = {}

    for (department, year), group in grouped:
        daily_counts = group.groupby('Day').size()
        
        # Determine the relevant days for the schedule
        if is_midterm:
            if year in [1, 3]:
                relevant_days = range(1, 6)  # First half (Days 0-4)
            else:
                relevant_days = range(6, 11) # Second half (Days 5-9)
        else:
            num_days = df['Day'].max()
            relevant_days = range(1, num_days+1)
        
        # Reindex to ensure all days are present, filling missing with 0
        full_daily_counts = daily_counts.reindex(list(relevant_days), fill_value=0)
            
        daily_counts_dict[(department, year)] = full_daily_counts.to_list()
        
    return daily_counts_dict


def analysis(experiment_no: int, is_midterm: bool, num_days: int, slots_per_day: int):
    # ... (your existing setup code for reading courses, rooms, etc.)
    course_xlsx = "./data/BerkData2.xlsx"
    room_xlsx = "./data/New Microsoft Excel Worksheet.xlsx"

    Course.read_courses(course_xlsx, is_midterm)
    Room.read_classroom_data(room_xlsx, num_days, slots_per_day, is_midterm)

    off_by_day = [[4] for _ in range(num_days)]
    off_by_day[4] = off_by_day[4] + [5]
    if num_days >= 10:
        off_by_day[9] = off_by_day[9] + [5]

    TimeSlot.generate_week(num_days, slots_per_day, off_by_day)
    horizon = num_days * slots_per_day

    exam = "midterms" if is_midterm else "finals"
    fac_df_manuel, fac_df_out = read_fac_xlsxs(experiment_no, exam)
    print(f"Manual Schedule Entries: {len(fac_df_manuel)}, Automated Schedule Entries: {len(fac_df_out)}")
    print("-" * 30)

    print("--- Room Utilization Analysis (Automated) ---")
    act_ts_au, util_au = room_utilization(fac_df_out)
    print("\n--- Room Utilization Analysis (Manual) ---")
    act_ts_ma, util_ma = room_utilization(fac_df_manuel, is_manuel=True)
    print("-" * 30)

    room_time_improvement = 100 - (act_ts_au / act_ts_ma) * 100
    room_utilization_improvement = 100 - (util_ma / util_au) * 100

    print(f"\nRoom-Time Improvement: {room_time_improvement:.2f}%")
    print(f"Room-Utilization Improvement: {room_utilization_improvement:.2f}%")
    print("-" * 30)

    # --- New Analysis for Exam Distribution ---
    print("\n--- Exam Distribution Analysis ---")
    dist_manual = analyze_exam_distribution(fac_df_manuel, is_midterm)
    dist_auto = analyze_exam_distribution(fac_df_out, is_midterm)

    avg_std_manual = np.mean(list(dist_manual.values()))
    avg_std_auto = np.mean(list(dist_auto.values()))
    
    distribution_improvement = 100 - (avg_std_auto / avg_std_manual) * 100

    print(f"Average Standard Deviation of Daily Exams (Manual): {avg_std_manual:.2f}")
    print(f"Average Standard Deviation of Daily Exams (Automated): {avg_std_auto:.2f}")
    print(f"Improvement in Exam Distribution Uniformity: {distribution_improvement:.2f}%")
    
    # You can also print the detailed distribution for each department-year if you want
    #print("\nDetailed Distribution (Manual):", dist_manual)
    #print("Detailed Distribution (Automated):", dist_auto)

        # --- New Analysis for Exam Distribution ---
    print("\n" + "="*40)
    print("--- Daily Exam Distribution Analysis ---")
    print("="*40)
    
    manual_counts = get_daily_exam_counts(fac_df_manuel, is_midterm)
    auto_counts = get_daily_exam_counts(fac_df_out, is_midterm)

    # Prepare data for a comparison DataFrame
    comparison_data = []
    all_keys = sorted(manual_counts.keys() | auto_counts.keys()) # Use union of keys

    for dept, year in all_keys:
        manual_dist = manual_counts.get((dept, year), 'N/A')
        auto_dist = auto_counts.get((dept, year), 'N/A')
        
        comparison_data.append({
            "Department": dept,
            "Year": year,
            "Manual": manual_dist,
            "Automated": auto_dist
        })
        
    # Create and display the DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Optional: Set pandas display options to see the full lists
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)

    # --- add 4 faculty-wide rows (sum across departments) ---

    def _elemwise_sum(series):
        lists = [x for x in series if isinstance(x, (list, tuple))]
        if not lists:
            return []
        maxlen = max(len(l) for l in lists)
        total = [0] * maxlen
        for l in lists:
            for i, v in enumerate(l):
                total[i] += int(v)
        return total

    faculty = (
        comparison_df
        .groupby('Year', sort=True)
        .agg({'Manual': _elemwise_sum, 'Automated': _elemwise_sum})
        .reset_index()
    )
    faculty.insert(0, 'Department', 'FACULTY')

    # append to your table
    comparison_df = pd.concat([comparison_df, faculty], ignore_index=True)

    # show just the 4 added rows
    print("\n=== FACULTY totals (per-day counts) ===")
    print(faculty.to_string(index=False))
    #print(comparison_df.to_string(index=False))
    





if __name__ == "__main__":
    runs_path = "./runs"
    os.makedirs(runs_path, exist_ok=True)
    experiment = len(os.listdir(runs_path))

    seed = None
    np.random.seed(seed)
    random.seed(seed)

    is_midterm = False
    num_days = 10 if is_midterm else 8  # midterm:10, final:8
    slots_per_day = 9
    #exam_scheduling_main(experiment, is_midterm, num_days, slots_per_day)
    analysis(19, is_midterm, num_days, slots_per_day)


