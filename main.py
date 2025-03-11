import numpy as np
import pandas as pd
import random
from ortools.sat.python import cp_model

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
        self.short = ""
        for word in words:
            self.short += word[0]
        Department.department_names.append(department_name)
        Department.departments.append(self)

    def add_course(self, course):
        if course not in self.courses:
            self.courses.append(course)
            self.curriculums.setdefault(course.year, []).append(course)

    @classmethod
    def get_department(cls, department_name: str):
        if department_name in cls.department_names:
            dep_idx = cls.department_names.index(department_name)
            return cls.departments[dep_idx]
        return cls(department_name)


class Course:
    course_list = []
    courses: pd.DataFrame
    course_codes = set()

    def __init__(
        self,
        id: int,
        department: Department,
        course_name: str,
        year: int,
        n_students: int,
        course_code: str,
        shared: bool,
        requires_lab: bool,
    ):
        self.id = int(id)
        self.n_students = int(n_students)
        self.departments = [department]
        self.course_code = course_code
        self.shared = shared
        self.requires_lab = requires_lab
        self.year = year
        self.course_name = course_name

        dep = self.departments[0]
        self.dep_short = dep.short

        self.blocks = []
        self.blocks = self.get_blocks()

    def get_blocks(self):
        """
        Blocks of two for courses
        """
        if len(self.blocks) == 0:
            n = 2
            blocks = []
            blocks.append(2)
            n -= 2
               
            self.blocks = blocks
        return self.blocks
    
    def get_duration(self):
        return self.get_blocks()[0]

    @staticmethod
    def read_courses(path: str):
        course_df = pd.read_excel(path, index_col=None, header=0)
        #print(course_df.head())
        for i, row in enumerate(course_df.values):
            department_name, \
            course_name, \
            year, \
            n_students, \
            course_code, \
            shared, \
            requires_lab = row
            
            dep = Department.get_department(department_name)
            if shared:
                if course_code in Course.course_codes:
                    course = Course.get_course(course_code)
                    course.add_department(dep, n_students)
                    dep.add_course(course)
                    continue

            new_course = Course(i, dep, course_name, year, n_students, course_code, shared == 1, requires_lab == 1)
            dep.add_course(new_course)
            Course.course_codes.add(course_code)
            Course.course_list.append(new_course)

        #Course.display()

    def add_department(self, dep: Department, n_students: int):
        self.departments.append(dep)
        self.n_students += n_students

    @staticmethod
    def get_course(course_code: str):
        for course in Course.course_list:
            if course.course_code == course_code:
                return course
        raise Exception(f"No course with {course_code} found.")

    @staticmethod
    def display():
        n = len(Course.course_list)
        dep_names = []
        course_names = []
        years = []
        n_students_list = []
        course_codes = []
        shares = []
        requires_labs = []

        for course in Course.course_list:
            department_name = ", ".join(dep.department_name for dep in course.departments)
            dep_names.append(department_name)

            course_names.append(course.course_name)
            years.append(course.year)
            n_students_list.append(course.n_students)
            course_codes.append(course.course_code)
            shares.append(course.shared)
            requires_labs.append(course.requires_lab)


        Course.courses = pd.DataFrame({
            "Department Names": dep_names,
            "Course Names": course_names,
            "Year": years,
            "Student Count": n_students_list,
            "Course Code": course_codes,
            "Shared": shares,
            "Lab": requires_labs,
        })

        print(f"department amount: {len(Department.departments)}")
        print(f"exam amount: {len(Course.course_list)}")
        print(f"n_students: {sum(n_students_list)}")
        print(Course.courses.head())


class TimeSlot:
    slot_list: list = list()
    day: pd.DataFrame
    slot_array: np.ndarray
    ids: np.ndarray
    offs: np.ndarray

    def __init__(self, id: int, is_off: bool, course_index: int = None, cr_index: int = None):
        self.id = int(id)
        self.is_off = bool(is_off)
        self.course_index = course_index
        self.cr_index = cr_index


    def generate_week(n_days: int, n_slots_per_day: int, off_slot_lists_per_day):
        total_slots = n_days * n_slots_per_day
        TimeSlot.ids = np.arange(total_slots)
        TimeSlot.offs = np.zeros(total_slots, dtype=bool)

        for day in range(n_days):
            off_slots = off_slot_lists_per_day[day]
            # Convert from day-based slot to global index
            for s in off_slots:
                global_idx = day * n_slots_per_day + s
                TimeSlot.offs[global_idx] = True

        TimeSlot.slot_array = np.column_stack([TimeSlot.ids, TimeSlot.offs])
        TimeSlot.slot_list = [
            TimeSlot(id=idx, is_off=off)
            for idx, off in zip(TimeSlot.ids, TimeSlot.offs)
        ]
        TimeSlot.day = pd.DataFrame(TimeSlot.slot_array, columns=["id", "is_off"])
        #TimeSlot.display()

    @staticmethod
    def display():
        print(TimeSlot.day)


class Room:
    rooms: pd.DataFrame
    room_list: list = list()
    ids: np.ndarray
    capacities: np.ndarray
    room_array: np.ndarray

    def __init__(
            self,
            id: int,
            room_code: str,
            capacity: int,
            c_type: str,
    ):
        self.id = int(id)
        self.capacity = int(capacity)
        self.room_code = room_code
        self.c_type = c_type
        self.is_lab = c_type == "Lab"
        if self.is_lab:
            short = "L"
        else:
            short = "D"
        self.shorthand = short

    @staticmethod
    def read_classroom_data(path: str):
        classroom_df = pd.read_excel(path)
        classroom_df = classroom_df[classroom_df["Room"].notna()][["Room", "Capacity", "Type"]]

        room_caps = []
        room_codes = []
        room_ids = []
        labs = []
        for i, row in enumerate(classroom_df.values):
            room_code, capacity, c_type = row
            room_ids.append(i)
            room_caps.append(capacity)
            room_codes.append(room_code)

            room = Room(i, room_code, capacity, c_type)
            if room.is_lab:
                labs.append(room)
            else:
                Room.room_list.append(room)
        Room.room_list += labs

        Room.room_array = np.column_stack([room_ids, room_codes, room_caps])
        Room.rooms = pd.DataFrame({
            "id": room_ids,
            "room_codes": room_codes,
            "capacities": room_caps
        })


    def display():
        print(Room.rooms)
    

from collections import defaultdict

def build_timetable(courses, rooms, horizon, solver, start_vars, is_in_room_vars, n_days):

    room_ids = [r.id for r in rooms]
    timetable = pd.DataFrame("", index=range(horizon), columns=room_ids)
    unique_rooms = defaultdict(list)

    for c in courses:
        for i, blk_size in enumerate(c.get_blocks):
            start_val = solver.Value(start_vars[(c.id, i)])

            for r in rooms:
                if solver.Value(is_in_room_vars[(c.id, i, r.id)]) == 1:
                    unique_rooms[r.id].append(blk_size)
 
                    for t in range(start_val, start_val + blk_size):
                        timetable.at[t, r.id] = f"C{c.id}({c.year}:{c.n_students}:{c.teacher.id})"

    print(f"Unique Rooms: {len(unique_rooms)}")
    times_used = []
    total_block_usage = []
    room_usage = {}
    for id, block_lengths in unique_rooms.items():
        times_used.append(len(block_lengths))
        total_block_usage.append(sum(block_lengths))
    
    print(f"number of rooms used: {len(times_used)}")
    print(f"sum times used: {(sum(times_used))}")
    print(f"sum total block usage: {(sum(total_block_usage))}")

    import datetime
    time_indexes = [datetime.time(h+8, 30).strftime("%H:%M") for h in list(range(9))]
    day_length = horizon // n_days  # e.g., if horizon=63 for 7 days, day_length=9
    day_tables = []


    for d in range(n_days):
        start_row = d * day_length
        end_row   = (d + 1) * day_length
        day_df = timetable.iloc[start_row:end_row, :].copy()
        day_df.index = time_indexes
        day_df.columns = [f"Rm{rooms[i].id}({rooms[i].capacity}[F{rooms[i].faculty_id}])" for i in range(len(rooms))]
        day_tables.append(day_df)
        
    
    # day_tables[d] is the timetable slice for day d
    return day_tables

def build_timetable2(courses, rooms, horizon, n_days, solver, start_vars, is_in_room_vars, seat):
    import datetime
    import pandas as pd
    
    room_ids = [r.id for r in rooms]
    timetable = pd.DataFrame("", index=range(horizon), columns=room_ids)

    for e in courses:
        start_val = solver.Value(start_vars[e.id])
        duration = e.get_duration()
        for t in range(start_val, start_val + duration):
            for r in rooms:
                in_room = solver.Value(is_in_room_vars[(e.id, r.id)])
                
                if in_room == 1:
                    current_val = timetable.at[t, r.id]
                    # Append info about how many seats are used
                    exam_info = f"{e.dep_short} {e.course_code}({e.n_students}:{e.year})"
                    
                    if current_val:  # cell isn't empty
                        timetable.at[t, r.id] = current_val + "|" + exam_info
                    else:
                        timetable.at[t, r.id] = exam_info


    time_indexes = [
        datetime.time(h + 8, 30).strftime("%H:%M") for h in range(9)
    ]
    day_length = horizon // n_days  # e.g., if horizon=63 for 7 days, day_length=9

    """ day_names = ["Pazartesi", "Sali", "Carsamba", "Persembe", "Cuma"]
    if n_days == 6:
        day_names = day_names + ["Cumartesi"] """
    day_names = ["Day - " + str(i) for i in range(1, n_days + 1)]

    day_tables = []
    all_days = []
    for d in range(n_days):
        start_row = d * day_length
        end_row   = (d + 1) * day_length
        day_df = timetable.iloc[start_row:end_row, :].copy()

        day_df.index = time_indexes
        day_df.columns = [
            f"{rooms[i].room_code}({rooms[i].capacity})"
            for i in range(len(rooms))
        ]
        day_df['Day'] = day_names[d]
        # Store as a tuple (day_name, dataframe)
        all_days.append(day_df)
        day_tables.append((day_names[d], day_df))
    
    combined_df = pd.concat(all_days)
    combined_df.to_excel("exam_schedule.xlsx")
    #excelify(combined_df)
    return day_tables

def excelify(dep_list: list, output_filename="exam_schedule.xlsx"):
    import pandas as pd
    import numpy as np

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
    
    workbook  = writer.book
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
    
    # Define a format for alternate day rows (faint yellow) with center alignment (if desired)
    # (You can remove 'align' and 'valign' if you want no alignment.)
    yellow_format = workbook.add_format({'bg_color': '#FFFFE0', 'align': 'center', 'valign': 'vcenter'})
    
    for i, row in combined_df.iterrows():
        day = row["Day"]
        if day != current_day:
            if current_day is not None:
                first_row = header_rows + start_idx
                last_row  = header_rows + i - 1
                worksheet.merge_range(first_row, 0, last_row, 0, current_day)
                if group_index % 2 == 0:
                    for r_idx in range(first_row, last_row + 1):
                        worksheet.set_row(r_idx, None, yellow_format)
                group_index += 1
            current_day = day
            start_idx = i
    if current_day is not None:
        first_row = header_rows + start_idx
        last_row  = header_rows + i
        worksheet.merge_range(first_row, 0, last_row, 0, current_day)
        if group_index % 2 == 0:
            for r_idx in range(first_row, last_row + 1):
                worksheet.set_row(r_idx, None, yellow_format)
    
    # --------------------------
    # Department coloring setup:
    # --------------------------
    # Define a list of colors for departments.
    color_list = ["#A9A9A9", "#800080", "#FFA500", "#FFFF00", "#FF0000", "#008000", "#ADD8E6"]
    # Build a mapping from department short to a color.
    dept_colors = {}
    for i, dep in enumerate(dep_list):
        dept_colors[dep.short] = color_list[i % len(color_list)]
    
    # Create a format for each department (only background color, no centering).
    dept_formats = {}
    for dep_short, color in dept_colors.items():
        dept_formats[dep_short] = workbook.add_format({'bg_color': color})
    
    # Define a default format (if department info is not found).
    default_format = workbook.add_format({})
    
    # Initialize a dictionary to count the total number of colored cells for each department.
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
                
                # If we got a valid department short and cell_format is not default,
                # update the count.
                if dep_short is not None and cell_format != default_format:
                    colored_counts[dep_short] += merged_length
                
                if end_merge > start_merge:
                    worksheet.merge_range(start_merge, col, end_merge, col, current_val, cell_format)
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


def get_off_chunks(slot_list):
    """
    slot_list: a list of TimeSlot objects with attributes `.id` and `.is_off`.
    Returns a list of tuples (start, end) where each chunk of consecutive off-slots
    goes from 'start' to 'end' in discrete time.
    """
    off_chunks = []
    current_chunk = []
    
    # Sort slot_list by their .id just in case
    sorted_slots = sorted(slot_list, key=lambda s: s.id)
    
    for slot in sorted_slots:
        if slot.is_off:
            current_chunk.append(slot.id)
        else:
            # If we just ended a chunk, finalize it
            if len(current_chunk) > 0:
                first = current_chunk[0]
                last = current_chunk[-1]
                # chunk covers [first .. last+1)
                off_chunks.append((first, last+1))
                current_chunk = []
    
    # If there's a chunk at the very end
    if len(current_chunk) > 0:
        first = current_chunk[0]
        last = current_chunk[-1]
        off_chunks.append((first, last+1))
    
    return off_chunks

def main_multi_day():
    num_days = 5
    slots_per_day = 9
    year = 4
    n_teachers = 9

    off_by_day = [[4] for _ in range(num_days)]
    off_by_day[-1] = off_by_day[-1] + [5]

    TimeSlot.generate_week(num_days, slots_per_day, off_by_day)
    Room.generate_rooms(((4, 48), (3, 36), (2, 72), (1, 90)), year)
    Teacher.generate_teachers(n_teachers)
    Course.generate_courses(28, (2,7), (15,90), Teacher.teachers)

    T = TimeSlot
    C = Course
    R = Room
    P = Teacher

    C_subsets = Course.years_subsets
    print(f"course_ids: {[c.id for subset in C_subsets for c in subset]}")
    print(f"C_subsets: {[c.year for subset in C_subsets for c in subset]}")

    model = cp_model.CpModel()
    horizon = num_days * slots_per_day

    all_intervals_per_room = {r.id: [] for r in Room.room_list}
    start_vars = {}
    is_in_rooms = {}


    all_intervals_for_subset = [[] for _ in C_subsets]
    all_intervals_for_teachers = [[] for _ in range(n_teachers)]
    seat_utilization_terms = [] # Obj func
    for c in Course.course_list:
        for i, block_size in enumerate(c.blocks):

            start_var = model.NewIntVar(0, horizon - block_size, f"start_c{c.id}_b{i}")
            end_var = model.NewIntVar(0, horizon, f"end_c{c.id}_b{i}")


            """
            *   1. Course Block and Interval Definition
            Block Duration Constraint
            For eachg course block, this constraint ensures that the end time is exactly the
            start time plus the block'S fixed duration. In other word, every scheduled block of
            a course must have the correct length
            """
            model.Add(end_var == start_var + block_size)
            
            """
            *   1. Course Block and Interval Definition
            Interval Variable Creation
            An interval variable is created for eachg course block. This varialbe represents the time
            span during which the course block takes place.
            """
            interval_var = model.NewIntervalVar(start_var, block_size, end_var,
                                                f"interval_c{c.id}_b{i}")
            
            c_subset_index = find_subset_of_course(c, C_subsets)
            all_intervals_for_subset[c_subset_index].append(interval_var)
            all_intervals_for_teachers[c.teacher.id].append(interval_var)

            in_room_bools = []
            for r in Room.room_list:
                in_r = model.NewBoolVar(f"inRoom_c{c.id}_b{i}_r{r.id}")

                """
                *   2. Room Assignment Constraints
                Capacity Check for Room Assignment Constraint
                For each room option, if a course block is assigned to that room (i.e. the boolean variable
                in_r is true), then the room's capacity must be at least as large as the number of students
                enrolled in the course
                """
                model.Add(r.capacity >= c.n_students).OnlyEnforceIf(in_r) # Encorce capacity(r) > n_students(c)
                in_room_bools.append(in_r)
                is_in_rooms[(c.id, i, r.id)] = in_r

                """
                *   2. Room Assignment Constraints
                Optional Interval per Room
                For each room, an optional interval is created that is "active" only when the course block is actually
                assigned to that room. These optional intervals will later be used to enforce that no two events in the
                same room overlap
                """
                opt_interval = model.NewOptionalIntervalVar(start_var, block_size, end_var,
                                                            in_r, f"optinterval_c{c.id}_b{i}_r{r.id}")
                all_intervals_per_room[r.id].append(opt_interval)

                """
                *   2. Room Assignment Constraints
                Objective Tern for Seat Utilization
                If a course block iis assigned to a room, this term adds the difference between the room's capacity and
                the number of students to the objective. Minimizing this term encourages assignments where the room's
                capacity closely fits the class size (i.e. reduces wasted space)
                """
                wasted_capacity = r.capacity - c.n_students
                seat_utilization_terms.append(is_in_rooms[(c.id, i, r.id)] * wasted_capacity)
                #wasted_ration = c.n_students / r.capacity
                #seat_utilization_terms.append(is_in_rooms[(c.id, i, r.id)] * wasted_ration)

            """
            *   2. Room Assignment Constraints
            Exactly-One Room Assignment Constraint
            For every course block, exactly one of the room assignment booleans must be true.
            This ensures that each block is scheduled in one-and only one- room
            """
            model.Add(sum(in_room_bools) == 1)
            start_vars[(c.id, i)] = start_var


    #model.Minimize(sum(seat_utilization_terms)) # Objective function
    off_chunks = get_off_chunks(TimeSlot.slot_list)

    day_vars = {}
    for c in C.course_list:
        for i, block_size in enumerate(c.get_blocks):

            # day_c_i in [0.. num_days - 1]
            day_c_i = model.NewIntVar(0, num_days - 1, f"day_c{c.id}_b{i}")
            day_vars[(c.id, i)] = day_c_i

            """
            *   3. Day and Time Slot Constraints
            Day Calculation from Start Time constraint
            The day on which a courseblock is scheduled is calculated by dividing its start time by the
            number of slots per day. This tells you on which dayt (0 to num_days - 1) the block begins
            """
            model.AddDivisionEquality(day_c_i, start_vars[(c.id, i)], slots_per_day)

            """
            *   3. Day and Time Slot Constraints
            Enforcing a Block Withing a Day constraint
            This constraint makes sure that a course block does not "spill over" into the next day by
            ensuring that the block finisihes before the day's end
            """
            model.Add(start_vars[(c.id, i)] + block_size <= (day_c_i + 1) * slots_per_day)
    
    for c in C.course_list:
        for d in range(num_days):
            blocks_in_day_lits = []
            for i, block_size in enumerate(c.get_blocks):
                day_c_i_d = model.NewBoolVar(f"day_c{c.id}_b{i}_d{d}")

                model.Add(day_vars[(c.id, i)] == d).OnlyEnforceIf(day_c_i_d)
                model.Add(day_vars[(c.id, i)] != d).OnlyEnforceIf(day_c_i_d.Not())

                blocks_in_day_lits.append(day_c_i_d)

            """
            *   3. Day and Time Slot Constraints
            At most One Block per Day per Course
            For every course, this constraint ensures that at most one block is scheduled on any given day
            (The booleans day_c_i_d are used to indicate whether a block is on day d)
            """    
            model.Add(sum(blocks_in_day_lits) <= 1)


    room_used = {r.id: model.NewBoolVar(f"room_used_{r.id}") for r in Room.room_list}
    for r in R.room_list:
        assignments_in_room = []
        for c in Course.course_list:
            for i, blk_size in enumerate(c.blocks):
                key = (c.id, i, r.id)
                if key in is_in_rooms:
                    assignments_in_room.append(is_in_rooms[key])
        """
        *   4. Room Usage and Overall Utilization
        Room Usage Indicator
        For each room, a boolean variable room_used is set to 1 if any course block is assigned there.
        This is later used in the objective funcion to help minimize the number of rooms used.
        """
        model.AddMaxEquality(room_used[r.id], assignments_in_room)
    
    """
    *   4. Room Usage and Overall Utilization
    Objective Term for Room Usage
    The sum of these terms (multiplied by a weight) is included in the objective to favor solutions that use fewer rooms
    """
    room_usage_terms = [room_used[r.id] for r in Room.room_list]

    central_x = {}
    central_y = {}
    for s, subset in enumerate(C_subsets):
        for d in range(num_days):
            """
            *   5. Location Centrality and Distance Terms
            Central Coordiantes for Course Subsets
            For each course subset (for example, courses from the same year) and foır each day,
            these variables represents a "central" room location where you'd like the courses to cluster
            """
            central_x[(s, d)] = model.NewIntVar(0, 100, f"central_x_subset{s}_day{d}")
            central_y[(s, d)] = model.NewIntVar(0, 100, f"central_y_subset{s}_day{d}")
    
    # Come back hea
    abs_diff_x = {}
    abs_diff_y = {}
    for s, subset in enumerate(C_subsets):
        for d in range(num_days):
            for c in subset:
                abs_diff_x[(c.id, s, d)] = model.NewIntVar(0, 100, f"abs_diff_x_c{c.id}_sub{s}_day{d}")
                abs_diff_y[(c.id, s, d)] = model.NewIntVar(0, 100, f"abs_diff_y_c{c.id}_sub{s}_day{d}")
                # constraints
                for b, _ in enumerate(c.blocks):
                    for r in R.room_list:
                        """
                        *   5. Location Centrality and Distance Terms
                        Absolute Difference Between Room Location and Central Point
                        For every course in the subset, these constraints compute the absolute difference between
                        the room where the course block is scheduled and the central ocation for that subset on that day.
                        These differences are later summed to encourage courses to be geographically close to each other
                        """
                        model.Add(abs_diff_x[(c.id, s, d)] >= r.x * is_in_rooms[(c.id, b, r.id)] - central_x[(s, d)])
                        model.Add(abs_diff_x[(c.id, s, d)] >= central_x[(s, d)] - r.x * is_in_rooms[(c.id, b, r.id)])
                        model.Add(abs_diff_y[(c.id, s, d)] >= r.y * is_in_rooms[(c.id, b, r.id)] - central_y[(s, d)])
                        model.Add(abs_diff_y[(c.id, s, d)] >= central_y[(s, d)] - r.y * is_in_rooms[(c.id, b, r.id)])

    distance_terms = []
    for s, subset in enumerate(C_subsets):
        for d in range(num_days):
            for c in subset:
                distance_terms.append(abs_diff_x[(c.id, s, d)] + abs_diff_y[(c.id, s, d)])
    #model.Minimize(sum(distance_terms))

    ### Minimize idle between subset c in day d
    earliest = {}
    latest = {}
    for s, subset in enumerate(C_subsets):
        for d in range(num_days):
            earliest[(s, d)] = model.NewIntVar(0, horizon, f"earliest_sub{s}_day{d}")
            latest[(s, d)] = model.NewIntVar(0, horizon, f"latest_sub{s}_day{d}")

    for s, subset in enumerate(C_subsets):
        for d in range(num_days):
            for c in subset:
                for b, block_size in enumerate(c.blocks):
                    """
                    *   6. Idle Time Minimization
                    Earliest and Latest Block Times per Subset per Day
                    For each subset of courses on each day, these constraints record the earliest starting time
                    and the latest finishing time among all blocks
                    """
                    model.Add(earliest[(s, d)] <= start_vars[(c.id, b)])
                    model.Add(latest[(s, d)] >= start_vars[(c.id, b)] + block_size)
    
    span = {}
    for s, subset in enumerate(C_subsets):
        for d in range(num_days):
            span[(s, d)] = model.NewIntVar(0, horizon, f"span_sub{s}_day{d}")

            """
            *   6. Idle Time Minimization
            Span Calculation (Idle Time Indicator)
            The "span" (the total time window during which the subset's classes occur) is computed.
            Minimizing this span would encourage classes for the same subset to be scheduled closer together,
            therby reducing idle gaps 
            """
            model.Add(span[(s, d)] == latest[(s, d)] - earliest[(s, d)])
    span_terms = [span[(s,d)] for s in range(len(C_subsets)) for d in range(num_days)]

    idle_time_weight = 2.0
    seat_utilization_weight = 8.0
    room_usage_weight = 2.0
    distance_weight = 5.0

    """
    *   9. Objective Function
    Combined Objective
    Room Usage Term: Penalizes using many rooms by adding a cost for each room that is used
    Seat Utilization Term: Penalizes wasted capacity, encouraging courses to be assigned to rooms
    that fit them well. 
    """
    objective_expression = room_usage_weight * sum(room_usage_terms)
    objective_expression += seat_utilization_weight * sum(seat_utilization_terms)
    #objective_expression += distance_weight * sum(distance_terms)
    #objective_expression += idle_time_weight * sum(span_terms)
    model.Minimize(objective_expression) #Objective function
    #objective_expression = sum(room_usage_terms)
    #model.Minimize(sum(seat_utilization_terms)) #Objective function
    #model.Minimize(sum(room_used[r.id] for r in Room.room_list)*0.8) #Objective function
    

    for r in Room.room_list:
        for (start_off, end_off) in off_chunks:
            """
            *   7. Unavailable Time ("Off") Periods for Rooms
            Adding Off-Chunksas Intervals 
            For each room, certain time chunks are defined as "off" or unavailable. These intervals
            are added to the room's list of intervals so thast no course block can be scheduled during these times
            """
            off_int = model.NewIntervalVar(start_off, end_off - start_off, end_off,
                                           f"unavail_r{r.id}_{start_off}_{end_off}")
            all_intervals_per_room[r.id].append(off_int)

    for r_id, intervals in all_intervals_per_room.items():
        """
        *   8. No-Overlap Constraints
        Room No-Overlap
        For each room, all interavals ( both course blocks anmd the off periods) must not overlap.
        This ensures that no two events are scheduled in the same room at the same time. 
        """
        model.AddNoOverlap(intervals)

    for i in range(len(C_subsets)):
        """
        *   8. No-Overlap Constraints
        Subset No-Overlap
        For each subset of courases (e.g.by year), this constraint prevents overlapping
        intervals among the courses in that subset. This might be used to avoid conflicts
        for students who share the same cirriculum

        """
        model.AddNoOverlap(all_intervals_for_subset[i])

    for p in P.ids:
        """
        *   8. No-Overlap Constraints
        Teacher No-Overlap
        For each teacher, the model enforces that the teacher's assigned course intervals do not overlap,
        ensuring that a teacher isn't scheduled to be in two places at once.
        """
        model.AddNoOverlap(all_intervals_for_teachers[p])

    

    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    """
        *   Course blocks have the proper length and are scheduled within day boundaries
        *   Each course block is assigned exactly one room that meets its capacity requirements
        *   Courses (especially those in the same subset) and teachers do not have overlapping time slots.
        *   Rooms are not double-booked and respect unavailable ("off") times.
        *   The schedule aims to minimize wasted room capacity and potentially minimizes distance and idle times.   
    """

    """ for c in C.course_list:
        for i, blk_size in enumerate(c.get_blocks):
            s_val = solver.Value(start_vars[(c.id, i)])
            d_val = solver.Value(day_vars[(c.id, i)])
            print(f"Course={c.id}, block={i}, start={s_val}, day={d_val}, subset={find_subset_of_course(c, C_subsets)}, teacher: {c.teacher.id}") """

    print(f"111 Number lessons: {C.courses['n_lessons'].sum()}")
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Solution found (status={}):".format(status))

        timetable_df = build_timetable(
            courses=C.course_list,
            rooms=R.room_list,
            horizon=len(T.slot_list),
            solver=solver,
            start_vars=start_vars,
            is_in_room_vars=is_in_rooms,
            n_days=num_days
        )
        for df in timetable_df:
            print(df)
            print()
    else:
        print("No solution found (status={}).".format(status))

def mission_report(solver, start, slots_per_day, in_room):
    # ---------------------------
    # 1. Count Total Exams Scheduled Per Day
    # ---------------------------
    exams_per_day = {}  # Mapping: day index -> count of exams
    for e in Course.course_list:
        exam_start = solver.Value(start[e.id])
        exam_day = exam_start // slots_per_day  # Day index (e.g. 0 for the first day)
        exams_per_day[exam_day] = exams_per_day.get(exam_day, 0) + 1

    print("Exams scheduled per day:")
    total_exam_amount = 0
    for day, count in sorted(exams_per_day.items()):
        print(f"  Day {day}: {count} exams")
        total_exam_amount += count

    print(f"Total exams: {total_exam_amount}")    
    # ---------------------------
    # 2. Count Rooms Used Per Day
    # ---------------------------
    rooms_used_per_day = {}  # Mapping: day index -> set of room IDs used on that day
    for e in Course.course_list:
        exam_start = solver.Value(start[e.id])
        exam_day = exam_start // slots_per_day
        if exam_day not in rooms_used_per_day:
            rooms_used_per_day[exam_day] = set()
        for r in Room.room_list:
            if solver.Value(in_room[(e.id, r.id)]) == 1:
                rooms_used_per_day[exam_day].add(r.id)
    
    print("Rooms used per day:")
    for day in sorted(rooms_used_per_day.keys()):
        print(f"  Day {day}: {len(rooms_used_per_day[day])} rooms used")
    
    # ---------------------------
    # 3. Count Exams per Department-Year per Day
    # ---------------------------
    # We'll assume each exam e has attributes e.department.id and e.year.
    dep_year_per_day = {}  # Key: (day, department_id, year), Value: count of exams
    for e in Course.course_list:
        exam_start = solver.Value(start[e.id])
        exam_day = exam_start // slots_per_day
        # For each exam, build a key using its day, department id, and year.
        key = (exam_day, e.departments[0].id, e.year)
        dep_year_per_day[key] = dep_year_per_day.get(key, 0) + 1

    print("Department-Year exam count per day:")
    # Sort keys by day, then department id, then year.
    for (day, dep_id, year) in sorted(dep_year_per_day.keys()):
        count = dep_year_per_day[(day, dep_id, year)]
        print(f"  Day {day}, Department {dep_id}, Year {year}: {count} exams")


def exam_scheduling_main():
    # ---------------------------
    # Basic parameters and data loading
    # ---------------------------
    num_days = 10
    slots_per_day = 9

    # Read course and room data from Excel files.
    course_xlsx = "./data/fall2425_course_info - Copy.xlsx"
    room_xlsx = "./data/New Microsoft Excel Worksheet.xlsx"
    Course.read_courses(course_xlsx)
    Room.read_classroom_data(room_xlsx)
    
    # Off-time settings: e.g., during lunch or cleaning.
    # Here, off_by_day is a list with one sublist per day.
    off_by_day = [[4] for _ in range(num_days)]
    off_by_day[4] = off_by_day[4] + [5]  # For day 5, two off-slot indexes.
    off_by_day[9] = off_by_day[9] + [5]  # For day 5, two off-slot indexes.
    TimeSlot.generate_week(num_days, slots_per_day, off_by_day)

    # Total number of time slots available.
    horizon = num_days * slots_per_day
    # ---------------------------
    # Create the CP model
    # ---------------------------
    model = cp_model.CpModel()

    # ---------------------------
    # (0) Interval Variables for Exams
    # Each exam gets a mandatory interval (start, duration, end)
    # ---------------------------
    start = {}
    end = {}
    interval_var = {}
    for e in Course.course_list:
        if e.year == 1 or e.year == 3:
            # start time for exam e can be anywhere between 0 and (horizon - duration)
            start[e.id] = model.NewIntVar(0, horizon // 2 - e.get_duration(), f"start_e{e.id}")
            # end time is between 0 and horizon
            end[e.id] = model.NewIntVar(0, horizon // 2, f"end_e{e.id}")
        else:
            # start time for exam e can be anywhere between 0 and (horizon - duration)
            start[e.id] = model.NewIntVar(horizon // 2, horizon - e.get_duration(), f"start_e{e.id}")
            # end time is between 0 and horizon
            end[e.id] = model.NewIntVar(horizon // 2, horizon, f"end_e{e.id}")
        # Fix exam duration: end = start + duration
        model.Add(end[e.id] == start[e.id] + e.get_duration())
        # Create the mandatory interval variable for exam e
        interval_var[e.id] = model.NewIntervalVar(start[e.id], e.get_duration(), end[e.id], f"interval_e{e.id}")

    # ---------------------------
    # (1) Seat Allocation and Room Assignment Variables
    # For each exam e and room r, create:
    #   - seat[(e, r)]: number of seats allocated in room r for exam e.
    #   - in_room[(e, r)]: Boolean; 1 if exam e uses room r.
    # For non-lab rooms, limit seat allocation to half capacity.
    # ---------------------------
    seat = {}
    in_room = {}
    for e in Course.course_list:
        for r in Room.room_list:
            if not r.is_lab:
                seat[(e.id, r.id)] = model.NewIntVar(0, r.capacity // 2, f"seat_e{e.id}_r{r.id}")
                in_room[(e.id, r.id)] = model.NewBoolVar(f"in_room_e{e.id}_r{r.id}")
                # If the room is not used (in_room==0), then seat must be 0.
                model.Add(seat[(e.id, r.id)] <= (r.capacity // 2) * in_room[(e.id, r.id)])
            else:
                seat[(e.id, r.id)] = model.NewIntVar(0, r.capacity, f"seat_e{e.id}_r{r.id}")
                in_room[(e.id, r.id)] = model.NewBoolVar(f"in_room_e{e.id}_r{r.id}")
                model.Add(seat[(e.id, r.id)] <= r.capacity * in_room[(e.id, r.id)])

    # ---------------------------
    # (2) Full Seat Allocation Constraint
    # For each exam, ensure that the sum of seats assigned across all rooms equals the number of students.
    # ---------------------------
    for e in Course.course_list:
        model.Add(sum(seat[(e.id, r.id)] for r in Room.room_list) == e.n_students)

    # ---------------------------
    # (3) Room Scheduling: Optional Intervals & NoOverlap in Rooms
    # For each exam and room, create an optional interval variable that is active if the exam is assigned to that room.
    # Then, enforce that in each room, these optional intervals do not overlap.
    # ---------------------------
    opt_int_per_room = {r.id: [] for r in Room.room_list}
    for e in Course.course_list:
        for r in Room.room_list:
            opt_interval = model.NewOptionalIntervalVar(
                start[e.id],
                e.get_duration(),
                end[e.id],
                in_room[(e.id, r.id)],
                f"opt_interval_e{e.id}_r{r.id}"
            )
            opt_int_per_room[r.id].append(opt_interval)
    # Enforce no overlap for exams in each room.
    # Comment/Uncomment bellow for no-overlap in rooms
    """ for r in Room.room_list:
        model.AddNoOverlap(opt_int_per_room[r.id]) """

    # --) Overlapping exam capacity constraint for each room
    for r in Room.room_list:
        if r.is_lab:
            model.AddCumulative(
                intervals=[interval_var[e.id] for e in Course.course_list],
                demands=[seat[(e.id, r.id)] for e in Course.course_list],
                capacity=r.capacity
            )
        else:
            model.AddCumulative(
                intervals=[interval_var[e.id] for e in Course.course_list],
                demands=[seat[(e.id, r.id)] for e in Course.course_list],
                capacity=r.capacity // 2
            )
    # ---------------------------
    # (4) Department/Year Conflict Constraint
    # For each department and year, ensure that exams do not overlap
    # (e.g., to avoid scheduling conflicts for students in the same curriculum).
    # ---------------------------
    dept_year_intervals = {}
    for department in Department.departments:
        for year, year_courses in department.curriculums.items():
            intervals = [interval_var[e.id] for e in year_courses]
            dept_year_intervals[(department.id, year)] = intervals
    
    for key, intervals in dept_year_intervals.items():
        model.AddNoOverlap(intervals)

    # ---------------------------
    # (5) Room-Type Constraint
    # If an exam requires a lab, it must not be scheduled in a non-lab room.
    # Similarly, if an exam does not require a lab, it must not be scheduled in a lab.
    # ---------------------------
    for e in Course.course_list:
        if e.requires_lab:
            for r in Room.room_list:
                if not r.is_lab:
                    model.Add(seat[(e.id, r.id)] == 0)
        else:
            for r in Room.room_list:
                if r.is_lab:
                    model.Add(seat[(e.id, r.id)] == 0)

    # ---------------------------
    # (6) Big-M Linking Constraint
    # Ensure that if a room is marked as used for an exam (in_room == 1),
    # then at least one seat is allocated (seat >= 1).
    # ---------------------------
    for e in Course.course_list:
        for r in Room.room_list:
            model.Add(seat[(e.id, r.id)] >= 1).OnlyEnforceIf(in_room[(e.id, r.id)])
    
    # ---------------------------
    # (7) Day Constraints
    # Ensure that each exam is scheduled entirely within a single day.
    # Here, we introduce a variable day_c for each exam and force the exam's start and end to fall within that day's bounds.
    # ---------------------------
    for e in Course.course_list:
        day_c = model.NewIntVar(0, num_days - 1, f"day_c{e.id}")
        model.Add(start[e.id] >= day_c * slots_per_day)
        model.Add(start[e.id] + e.get_duration() <= (day_c + 1) * slots_per_day)
        
    # ---------------------------
    # (8) Off-Time Constraints
    # Prevent exams from being scheduled during times when rooms are unavailable.
    # For each room, create off intervals from the off_chunks (global off times),
    # then enforce that the off intervals do not overlap with the exam optional intervals.
    # ---------------------------
    off_chunks = get_off_chunks(TimeSlot.slot_list)
    off_intervals_per_room = {r.id: [] for r in Room.room_list}
    for r in Room.room_list:
        for (start_off, end_off) in off_chunks:
            off_int = model.NewIntervalVar(start_off, end_off - start_off, end_off,
                                           f"unavail_r{r.id}_{start_off}_{end_off}")
            off_intervals_per_room[r.id].append(off_int)
    # For each room, combine the off intervals with the exam intervals and enforce no overlap.
    for r in Room.room_list:
        model.AddNoOverlap(off_intervals_per_room[r.id] + opt_int_per_room[r.id])
    
    # ---------------------------
    # (9) Balanced Exam Distribution
    # ---------------------------
    """ local_day = {}
    week_length = num_days // 2

    for e in Course.course_list:
        local_day[e.id] = model.NewIntVar(0, week_length - 1, f"local_day_e{e.id}")
        if e.year in {1, 3}:
            model.AddDivisionEquality(local_day[e.id], start[e.id], slots_per_day)
        else:
            model.AddDivisionEquality(
                local_day[e.id],
                model.NewIntVarFromDomain(cp_model.Domain.FromIntervals(
                    [[slots_per_day * week_length, horizon - e.get_duration()]]
                ),
                f"second_week_{e.id}") - slots_per_day * week_length, slots_per_day
            )
    
    count = {}
    for department in Department.departments:
        for year, exams in department.curriculums.items():
            for i in range(week_length):
                count[(department.id, year, i)] = model.NewIntVar(0, len(exams),
                            f"count_dep{department.id}_year{year}_day{i}")

                indicators = []
                for e in exams:
                    indicator = model.NewBoolVar(f"exam_{e.id}_on_day{i}")
                    model.Add(local_day[e.id] == i).OnlyEnforceIf(indicator)
                    model.Add(local_day[e.id] != i).OnlyEnforceIf(indicator.Not())

                model.Add(count[(department.id, year, i)] == sum(indicators)) """

    """ tolarance = 1
    for department in Department.departments:
        for year, exams in department.curriculums.items():
            total_exams = len(exams)
            avg = total_exams / week_length
            for i in range(week_length):
                lower_bound = int(avg)
                upper_bound = int(avg) + tolarance
                model.Add(count[(department.id, year, i)] >= lower_bound)
                model.Add(count[(department.id, year, i)] <= upper_bound) """
    
    """ total_deviation = []
    for department in Department.departments:
        for year, exams in department.curriculums.items():
            total_exams = len(exams)
            target = total_exams / week_length
            for i in range(week_length):
                deviation = model.NewIntVar(0, week_length, f"dev_dep{department.id}_year{year}_day{i}")
                model.Add(deviation >= count[(department.id, year, i)] - int(target))
                model.Add(deviation >= int(target) - count[(department.id, year, i)])
                total_deviation.append(deviation) """
            

    # ---------------------------
    # (9) (Optional) Objective: Minimize total room usage.
    # This would encourage the solver to assign each exam to as few rooms as possible.
    # Uncomment if needed.
    """ total_room_usage = sum(in_room[(e.id, r.id)] for e in Course.course_list for r in Room.room_list)
    tru_weight = 5 """
    #model.Minimize(total_room_usage * tru_weight)
    # ---------------------------

    # ---------------------------
    # (10) (Optional) Objective: Minimize sum deviation to balance the exams in each day.
    # This would encourage the solver to assign each exam from the same cirriculums to evenly spread out thourgh the week.
    # Uncomment if needed.
    """ balanced_exams = sum(total_deviation)
    balanced_weight = 1
    all_objectives = balanced_exams * balanced_weight + total_room_usage * tru_weight """
    #model.Minimize(all_objectives)
    # ---------------------------

    
    # ---------------------------
    # Solve the model.
    # ---------------------------
    # Phase 1: Solve for a feasible solution without the optimization objective.
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 300
    solver.parameters.num_search_workers = 4
    status = solver.solve(model)
    
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if status == cp_model.OPTIMAL:
            print("optimal")
        else:
            print("feasible")
        # Build and print the timetable (build_timetable2 is assumed to exist)
        timetable_df = build_timetable2(Course.course_list, Room.room_list, horizon, num_days, solver, start, in_room, seat)
        mission_report(solver, start, slots_per_day, in_room)
        """ hints = []
        for e in Course.course_list:
            hints.append((start[e.id], solver.Value(start[e.id])))
            for r in Room.room_list:
                hints.append((seat[(e.id, r.id)], solver.Value(seat[(e.id, r.id)])))
                hints.append((in_room[(e.id, r.id)], solver.Value(in_room[(e.id, r.id)])))

        for var, value in hints:
            model.AddHint(var, value) """

        # Phase 2: Clear the previous objective and add your new objective.
        # model.ClearObjective()
        # model.Minimize(total_room_usage * tru_weight)

        # Add new constraint
        """ for r in Room.room_list:
            if r.is_lab:
                model.AddCumulative(
                    intervals=[interval_var[e.id] for e in Course.course_list],
                    demands=[seat[(e.id, r.id)] for e in Course.course_list],
                    capacity=r.capacity
                )
            else:
                model.AddCumulative(
                    intervals=[interval_var[e.id] for e in Course.course_list],
                    demands=[seat[(e.id, r.id)] for e in Course.course_list],
                    capacity=r.capacity // 2
                ) """

        """ # Re-solve with the new objective, using the hints.
        status = solver.solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            if status == cp_model.OPTIMAL:
                print("optimal")
            else:
                print("feasible")
            # Build and print the timetable (build_timetable2 is assumed to exist)
            timetable_df = build_timetable2(Course.course_list, Room.room_list, horizon, num_days, solver, start, in_room, seat)
            mission_report(solver, start, slots_per_day, in_room)

        else:
            print("No solution") """
    else:
        print("No solution")


if __name__ == "__main__":
    seed = None
    np.random.seed(seed)
    random.seed(seed)   

    #main_multi_day()
    exam_scheduling_main()
    course_xlsx = "./data/fall2425_course_info - Copy.xlsx"
    Course.read_courses(course_xlsx)
    for d in Department.departments:
        print(d.name, d.id)
    excelify(Department.departments)
