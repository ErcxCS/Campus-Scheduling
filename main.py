import numpy as np
import pandas as pd
import random
from ortools.sat.python import cp_model
from matplotlib import pyplot as plt


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

    @classmethod
    def get_department(cls, department_name: str):
        if department_name in cls.department_names:
            idx = cls.department_names.index(department_name)
            return cls.departments[idx]
        return cls(department_name)


class Course:
    course_list = []
    course_codes = set()

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
        # (This mirrors the old `get_blocks()` logic, which always returned [2].)
        self.blocks = [2]

    def get_duration(self):
        return self.blocks[0]

    @staticmethod
    def read_courses(path: str):
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

            dep = Department.get_department(department_name)
            # If a course with the same code and instructor already exists, just add the new department
            if course_code in Course.course_codes:
                existing = next(
                    (c for c in Course.course_list if c.course_code == course_code), None
                )
                if existing is not None and existing.instructor_id == instructor_id:
                    existing.departments.append(dep)
                    existing.n_students += int(n_students)
                    dep.add_course(existing)
                    continue

            new_course = Course(
                id=i,
                department=dep,
                course_name=course_name,
                year=int(year),
                n_students=int(n_students),
                course_code=course_code,
                instructor_id=int(instructor_id),
                requires_lab=(requires_lab == 1),
                mandatory=mandatory
            )
            dep.add_course(new_course)
            Course.course_codes.add(course_code)
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

    def __init__(self, id: int, room_code: str, capacity: int, c_type: str):
        self.id = int(id)
        self.capacity = int(capacity)
        self.room_code = room_code
        self.is_lab = (c_type == "Lab")

    @staticmethod
    def read_classroom_data(path: str):
        df = pd.read_excel(path)
        df = df[df["Room"].notna()][["Room", "Capacity", "Type"]]

        room_ids = []
        room_codes = []
        room_caps = []

        labs = []
        regulars = []
        for i, row in enumerate(df.values):
            room_code, capacity, c_type = row
            room_ids.append(i)
            room_caps.append(int(capacity))
            room_codes.append(room_code)

            room_obj = Room(i, room_code, int(capacity), c_type)
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


def build_timetable2(courses, rooms, horizon, n_days, solver, start_vars, in_room_vars, seat_vars):
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
    combined.to_excel("exam_schedule.xlsx", index=False)
    return all_days


def department_exam_schedule(departments, courses, rooms, horizon, n_days, solver, start_vars, in_room_vars):
    day_length = horizon // n_days
    import datetime

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
                        assigned_rooms += r.room_code + " - "

                data.append({
                    'Year:': year,
                    'Course ID': course.course_code,
                    'Course Name': course.course_name,
                    'Date': day_str,
                    'Starting Hours': date_time,
                    'Rooms': assigned_rooms
                })
        dep_df = pd.DataFrame(data)
        excel_name = dep.name + ".xlsx"
        dep_df.to_excel(excel_name, index=False)


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


def exam_scheduling_main():
    # ---------------------------
    # 0) Parameters & Data Loading
    # ---------------------------
    num_days = 8
    slots_per_day = 9

    course_xlsx = "./data/BerkData2.xlsx"
    room_xlsx = "./data/New Microsoft Excel Worksheet.xlsx"

    Course.read_courses(course_xlsx)
    Room.read_classroom_data(room_xlsx)

    # Off-time per day (e.g. lunch slot = 4)
    off_by_day = [[4] for _ in range(num_days)]
    off_by_day[4] = off_by_day[4] + [5]  # Day 5 has two off slots

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

        # Also enforce a “no more than 3 simultaneous exams” hard cap per room:
        model.AddCumulative(intervals=opt_int_per_room[r.id],
                            demands=[1] * len(Course.course_list),
                            capacity=3)

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

    # (7) Link “in_room ⇒ seat ≥ 1”
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

    # (11) “Mission active” helper → minimize how many room‐times are actually used
    # (Optional: we include it in the final objective)
    mission_active = {}
    for r in Room.room_list:
        for t in TimeSlot.slot_list:
            mission_active[(r.id, t.id)] = model.NewBoolVar(f"mission_active_r{r.id}_t{t.id}")
            active_list = []
            for e in Course.course_list:
                b_start = model.NewBoolVar(f"bstart_e{e.id}_before_{t.id}")
                b_end = model.NewBoolVar(f"bend_e{e.id}_after_{t.id}")

                # b_start ⇔ (start[e] <= t.id)
                model.Add(start[e.id] <= t.id).OnlyEnforceIf(b_start)
                model.Add(start[e.id] > t.id).OnlyEnforceIf(b_start.Not())

                # b_end ⇔ (end[e] > t.id)
                model.Add(end[e.id] > t.id).OnlyEnforceIf(b_end)
                model.Add(end[e.id] <= t.id).OnlyEnforceIf(b_end.Not())

                active_bool = model.NewBoolVar(f"active_e{e.id}_r{r.id}_t{t.id}")
                model.AddBoolAnd([b_start, b_end, in_room[(e.id, r.id)]]).OnlyEnforceIf(active_bool)
                model.AddBoolOr([b_start.Not(), b_end.Not(), in_room[(e.id, r.id)].Not()]) \
                     .OnlyEnforceIf(active_bool.Not())
                active_list.append(active_bool)

            model.AddMaxEquality(mission_active[(r.id, t.id)], active_list)

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
    solver.parameters.max_time_in_seconds = 600
    solver.parameters.num_search_workers = 16
    solver.parameters.log_search_progress = True

    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("Solution status:", "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE")
        # Build the Excel‐output timetable
        build_timetable2(Course.course_list, Room.room_list, horizon, num_days,
                         solver, start, in_room, seat)
        
        department_exam_schedule(Department.departments,
                                 Course.course_list,
                                 Room.room_list,
                                 horizon,
                                 num_days,
                                 solver,
                                 start,
                                 in_room)
        # Print and plot mission report
        mission_report(solver, start, slots_per_day, in_room, num_days)
    else:
        print("No solution found (status {}).".format(status))


def excelify(dep_list: list, output_filename="exam_schedule.xlsx"):
    import pandas as pd

    # Read the combined timetable DataFrame from the Excel file.
    combined_df = pd.read_excel(output_filename)

    # Ensure "Day" is the first column.
    cols = combined_df.columns.tolist()
    if "Day" in cols and cols[0] != "Day":
        cols.remove("Day")
        cols = ["Day"] + cols
        combined_df = combined_df[cols]

    # Write the DataFrame to Excel using XlsxWriter.
    writer = pd.ExcelWriter("EDITED2_" + output_filename, engine="xlsxwriter")
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

    # Print out the total number of colored cells for each department.
    print("Colored cells count by department:")
    total_mission = 0
    for dep_short, count in colored_counts.items():
        true_count = count // 2
        total_mission += true_count
        print(f"Department {dep_short}: {true_count} cells")

    print(f"Total mission count: {total_mission}")


if __name__ == "__main__":
    seed = None
    np.random.seed(seed)
    random.seed(seed)
    exam_scheduling_main()
    excelify(Department.departments)
